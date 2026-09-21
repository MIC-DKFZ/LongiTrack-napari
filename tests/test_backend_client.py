from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from longitrack_backend import server

from longitrack_napari.backend.client import BackendClient

# only the tests that actually run a model need one; the wire-protocol tests below don't
requires_model = pytest.mark.skipif(
    not os.environ.get("LONGITRACK_MODEL_DIR"),
    reason="set LONGITRACK_MODEL_DIR to a LongiSeg tracking model folder to run the backend tests",
)


@pytest.fixture(scope="module")
def client():
    backend = BackendClient()
    backend.ensure_started()
    backend.initialize(location=None, device="cuda")
    yield backend
    backend.shutdown()


def _assert_torch_free_subprocess(code: str, timeout: float) -> None:
    # a check against THIS process's sys.modules would be at the mercy of pytest's
    # collection order: it imports every test *file* up front, and test_export.py's
    # top-level `from longitrack_backend.inference import ...` pulls torch in before any
    # test body runs, no matter which file's tests execute first. A fresh subprocess is
    # the only way to make this claim regardless of what else is in the test suite.
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout
    )
    assert result.returncode == 0, f"subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


@requires_model
def test_the_gui_process_never_imports_torch():
    # this IS the point of the whole exercise: proven by checking a subprocess's own
    # sys.modules after it starts a real backend and initializes a real model, not by
    # reading the source and hoping nothing changed it.
    _assert_torch_free_subprocess(
        "import sys\n"
        "from longitrack_napari.backend.client import BackendClient\n"
        "client = BackendClient()\n"
        "client.ensure_started()\n"
        "client.initialize(location=None, device='cuda')\n"
        "client.shutdown()\n"
        "assert 'torch' not in sys.modules, sorted(sys.modules)\n",
        timeout=180,
    )


@requires_model
def test_track_end_to_end(client, sample_pair):
    bl_path, fu_path, bl_point, fu_point = sample_pair
    result = client.track(str(bl_path), bl_point, str(fu_path), fu_point, lesion_id="lesion-1", lesion_number=1)

    assert result["followup"]["mask"].shape == (64, 160, 160)
    assert result["followup"]["mask"].dtype == np.uint8
    assert result["followup"]["voxels"] > 0
    assert result["baseline"]["mask"].shape == (64, 160, 160)
    assert result["reused"] is False


@requires_model
def test_propagate_returns_a_point_inside_the_volume(client, sample_pair):
    bl_path, fu_path, bl_point, _ = sample_pair
    propagations = client.propagate(str(bl_path), str(fu_path), [bl_point], (64, 160, 160))

    assert len(propagations) == 1
    assert propagations[0].error is None
    followup_index = propagations[0].followup_index
    assert all(0 <= c < s for c, s in zip(followup_index, (64, 160, 160), strict=True))


@requires_model
def test_export_writes_files(client, sample_pair, tmp_path):
    bl_path, fu_path, bl_point, fu_point = sample_pair
    client.track(str(bl_path), bl_point, str(fu_path), fu_point, lesion_id="export-me", lesion_number=1)

    written = client.export(str(tmp_path), ["export-me"])
    assert len(written) == 3  # baseline mask, follow-up mask, inference_meta.json
    assert all(os.path.isfile(path) for path in written)
    assert any(path.endswith("inference_meta.json") for path in written)


@requires_model
def test_export_with_an_unknown_lesion_id_raises(client, tmp_path):
    from longitrack_napari.backend.client import BackendError

    with pytest.raises(BackendError):
        client.export(str(tmp_path), ["no-such-lesion"])


@pytest.fixture(scope="module")
def sample_pair():
    from synthetic import BASELINE_CENTER, FOLLOWUP_SHIFT, write_sample_pair

    bl_path, fu_path = write_sample_pair()
    bl_point = list(BASELINE_CENTER)
    fu_point = [c + s for c, s in zip(BASELINE_CENTER, FOLLOWUP_SHIFT, strict=True)]
    return bl_path, fu_path, bl_point, fu_point


# ------------------------------------------------------- wire protocol, no model needed -
def test_tcp_client_authenticates_with_an_authorized_ed25519_key(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "client.pem"
    private_path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    authorized_path = tmp_path / "authorized_keys"
    authorized_path.mkdir()
    (authorized_path / "client.pub").write_bytes(
        private.public_key().public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH) + b"\n"
    )
    backend = server.TcpBackendServer("127.0.0.1", 0, authorized_keys=server.load_authorized_keys(authorized_path))
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = backend.server_address
        client = BackendClient()
        client.connect_tcp(host, port, private_key=private_path)
        assert client.health()["service"] == "longitrack-backend"
    finally:
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=2)


def test_upload_scan_streams_to_a_content_addressed_backend_cache(tmp_path):
    source = tmp_path / "scan.nii.gz"
    source.write_bytes(b"scan-data" * 100_000)
    backend = server.TcpBackendServer("127.0.0.1", 0)
    backend.state.scan_cache_dir = tmp_path / "backend-cache"
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    try:
        client = BackendClient()
        client.connect_tcp(*backend.server_address)
        uploaded = Path(client.upload_scan(source))
        assert uploaded.read_bytes() == source.read_bytes()
        assert Path(client.upload_scan(source)) == uploaded
    finally:
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=2)


def test_kill_closes_any_active_sockets_too():
    # a GPU call that is genuinely in flight has an active socket; kill() must free the
    # waiting client thread the same way cancel_active() does, on top of restarting the
    # process itself
    left, right = socket.socketpair()
    client = BackendClient()
    client._active_sockets.add(left)
    client.kill()

    assert right.recv(1) == b""
    right.close()
