from __future__ import annotations

import socket
import socketserver
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from longitrack_napari.backend import server
from longitrack_napari.backend.client import BackendClient
from longitrack_napari.inference import _predict_cached_patch


def _result() -> SimpleNamespace:
    mask = SimpleNamespace(
        mask=np.zeros((2, 3, 4), dtype=np.uint8),
        point=[0, 0, 0],
        spacing_zyx=(1.0, 1.0, 1.0),
        scan=None,
        properties={},
        volume_ml=0.0,
        voxels=0,
    )
    return SimpleNamespace(followup=mask, baseline=None, seconds=0.0, notes=[])


def test_backend_servers_do_not_run_model_requests_concurrently():
    # File uploads may run concurrently with a model request, but the server protects
    # all model/registration state through this single operation lock.
    assert issubclass(server.UnixBackendServer, socketserver.ThreadingMixIn)
    assert issubclass(server.TcpBackendServer, socketserver.ThreadingMixIn)
    assert hasattr(server._State(), "operation_lock")


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


def test_clear_scans_drops_the_engine_and_registration_caches_not_the_model():
    class FakeEngine:
        def __init__(self):
            self.cache_cleared = False

        def clear_cache(self):
            self.cache_cleared = True

    class FakeRegistration:
        def __init__(self):
            self.prepared_scans_cleared = False

        def invalidate(self):
            pass

        def clear_prepared_scans(self):
            self.prepared_scans_cleared = True

    state = server._State()
    engine, registration = FakeEngine(), FakeRegistration()
    state.engine = engine
    state.registration = registration

    server.handle_invalidate_registration(state, {"clear_scans": True}, lambda _message: None)

    assert engine.cache_cleared
    assert registration.prepared_scans_cleared
    # the model/checkpoint itself is untouched: nothing here resets state.engine to None
    assert state.engine is engine


def test_track_cache_includes_baseline_option_and_file_signature(tmp_path):
    image = tmp_path / "image.nii.gz"
    image.write_bytes(b"image")

    class Engine:
        def __init__(self):
            self.calls = 0

        @staticmethod
        def _signature(path):
            stat = path.stat()
            return str(path), stat.st_mtime_ns, stat.st_size

        def track(self, *_args, **_kwargs):
            self.calls += 1
            return _result()

    state = server._State()
    state.engine = Engine()
    request = {
        "baseline_path": str(image),
        "followup_path": str(image),
        "baseline_point": [0, 0, 0],
        "followup_point": [0, 0, 0],
        "segment_baseline": False,
    }
    server.handle_track(state, request, lambda _message: None)
    request["segment_baseline"] = True
    server.handle_track(state, request, lambda _message: None)

    assert state.engine.calls == 2


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


def test_cached_patch_adapter_keeps_prediction_on_its_input_device():
    class Predictor:
        def __init__(self):
            self.list_of_parameters = [{}]
            self.network = torch.nn.Identity()
            self.device = torch.device("cpu")
            self.configuration_manager = SimpleNamespace(patch_size=(4, 4, 4))

        def _internal_maybe_mirror_and_predict(self, data):
            return torch.zeros((1, 2, *data.shape[2:]), dtype=torch.float32, device=data.device)

    prediction, lower, upper = _predict_cached_patch(
        torch.zeros((1, 5, 5, 5)),
        [2, 2, 2],
        torch.zeros((1, 5, 5, 5)),
        [2, 2, 2],
        Predictor(),
        [4, 4, 4],
        torch.device("cpu"),
        1.0,
    )

    assert prediction.shape == (2, 4, 4, 4)
    assert prediction.device.type == "cpu"
    assert lower == (0, 0, 0)
    assert upper == (4, 4, 4)


# ------------------------------------------------- GPU work stays on one thread -
def test_every_cuda_touching_request_is_routed_to_the_gpu_thread():
    # A thread's first forward pass pays a large per-thread CUDA setup cost, and it
    # explodes once both model stacks are resident (measured: ~21 s for the first
    # uniGradICON forward on a thread with LongiSeg loaded, ~0.4 s afterwards). The
    # server answers every request on a fresh connection thread, so anything that
    # touches torch has to be handed to the one long-lived GPU thread instead --
    # dropping a request type from this set silently reintroduces ~21 s per call.
    assert server._GPU_REQUESTS == {
        "initialize",
        "load_scan",
        "load_scans",
        "preload_registration_scans",
        "warm_up_registration",
        "release_scan",
        "propagate",
        "invalidate_registration",
        "track",
    }


def test_the_gpu_worker_runs_everything_on_one_thread_that_is_not_the_caller():
    import threading

    worker = server._GpuWorker()
    threads = {worker.run(lambda: threading.current_thread()) for _ in range(5)}

    assert len(threads) == 1, "every call must land on the same thread"
    assert threads.pop() is not threading.current_thread()

