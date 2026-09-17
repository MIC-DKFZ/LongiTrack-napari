import os
import subprocess
import sys

import numpy as np
import pytest

from longitrack_napari.backend import protocol as proto
from longitrack_napari.backend.client import BackendClient

pytestmark = pytest.mark.skipif(
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
    # top-level `from longitrack_napari.inference import ...` pulls torch in before any
    # test body runs, no matter which file's tests execute first. A fresh subprocess is
    # the only way to make this claim regardless of what else is in the test suite.
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout
    )
    assert result.returncode == 0, f"subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


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


def test_track_end_to_end(client, sample_pair):
    bl_path, fu_path, bl_point, fu_point = sample_pair
    result = client.track(str(bl_path), bl_point, str(fu_path), fu_point, lesion_id="lesion-1", lesion_number=1)

    assert result["followup"]["mask"].shape == (64, 160, 160)
    assert result["followup"]["mask"].dtype == np.uint8
    assert result["followup"]["voxels"] > 0
    assert result["baseline"]["mask"].shape == (64, 160, 160)
    assert result["reused"] is False


def test_propagate_returns_a_point_inside_the_volume(client, sample_pair):
    bl_path, fu_path, bl_point, _ = sample_pair
    propagations = client.propagate(str(bl_path), str(fu_path), [bl_point], (64, 160, 160), fast=True)

    assert len(propagations) == 1
    assert propagations[0].error is None
    followup_index = propagations[0].followup_index
    assert all(0 <= c < s for c, s in zip(followup_index, (64, 160, 160), strict=True))


def test_export_writes_files(client, sample_pair, tmp_path):
    bl_path, fu_path, bl_point, fu_point = sample_pair
    client.track(str(bl_path), bl_point, str(fu_path), fu_point, lesion_id="export-me", lesion_number=1)

    written = client.export(str(tmp_path), ["export-me"])
    assert len(written) == 3  # baseline mask, follow-up mask, inference_meta.json
    assert all(os.path.isfile(path) for path in written)
    assert any(path.endswith("inference_meta.json") for path in written)


def test_export_with_an_unknown_lesion_id_raises(client, tmp_path):
    from longitrack_napari.backend.client import BackendError

    with pytest.raises(BackendError):
        client.export(str(tmp_path), ["no-such-lesion"])


# ------------------------------------------------------------------- protocol --
def test_rle_round_trips_a_lesion_shaped_mask():
    mask = np.zeros((64, 160, 160), dtype=np.uint8)
    mask[20:30, 60:90, 70:100] = 1
    arrays: list = []
    packed = proto._pack({"m": mask}, arrays)

    assert packed["m"]["encoding"] == "rle"
    assert sum(a.nbytes for a in arrays) < mask.nbytes // 100  # a compact lesion compresses hugely

    restored = proto._unpack(packed, [a.tobytes() for a in arrays])["m"]
    assert np.array_equal(restored, mask)
    assert restored.dtype == mask.dtype



def unix_socket_pair():
    import socket

    a, b = socket.socketpair()
    yield a, b
    a.close()
    b.close()


@pytest.fixture(scope="module")
def sample_pair():
    from synthetic import BASELINE_CENTER, FOLLOWUP_SHIFT, write_sample_pair

    bl_path, fu_path = write_sample_pair()
    bl_point = list(BASELINE_CENTER)
    fu_point = [c + s for c, s in zip(BASELINE_CENTER, FOLLOWUP_SHIFT, strict=True)]
    return bl_path, fu_path, bl_point, fu_point


def test_resolve_device_falls_back_to_cpu_and_logs_why(monkeypatch):
    import torch

    from longitrack_napari.backend import server

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    messages = []
    assert server._resolve_device("cuda", messages.append) == "cpu"
    assert any("cpu" in m.lower() for m in messages)

