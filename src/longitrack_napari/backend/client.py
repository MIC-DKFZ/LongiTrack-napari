from __future__ import annotations

import atexit
import base64
import hashlib
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from longitrack_backend import protocol

Progress = Callable[[str], None]


class BackendError(RuntimeError):
    pass


def _die_with_parent() -> None:
    """Ask the kernel to SIGTERM this child when its parent goes away (Linux only).

    atexit cleanup is not enough: a hard-killed or crashed GUI/test process never runs
    it, and the backend then lives on holding a CUDA context and GPU memory.
    """
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - best effort, non-Linux or no libc
        pass


class BackendClient:
    # the GUI process's only connection to anything torch/CUDA/model-shaped.
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._socket_path: str | None = None
        self._host: tuple[str, int] | None = None
        self._tcp_private_key: Path | None = None
        self._session_id: str | None = None
        self._active_sockets: set[socket.socket] = set()

    # -------------------------------------------------------------- lifecycle --
    def ensure_started(self, progress: Progress | None = None) -> bool:
        """Ensure a local backend exists and return whether one was just started."""
        started = False
        with self._lock:
            if self._host is not None:
                return False
            if self._process is not None and self._process.poll() is None:
                return False
            self._discard_dead_process_locked()
            self._socket_path = os.path.join(tempfile.gettempdir(), f"longitrack-backend-{uuid.uuid4().hex}.sock")
            if progress:
                progress("Starting the model backend process...")
            self._process = subprocess.Popen(
                [sys.executable, "-m", "longitrack_backend.server", "--socket", self._socket_path],
                # A PIPE would need to be drained for the backend's entire lifetime or
                # verbose dependencies could eventually deadlock on a full pipe.
                stdout=subprocess.DEVNULL,
                stderr=None,
                preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None,
            )
            atexit.register(self._terminate)
            started = True
        try:
            self._wait_ready()
            self._start_session()
        except Exception:
            self._terminate()
            raise
        return started

    def connect_tcp(
        self, host: str, port: int, private_key: str | Path | None = None,
    ) -> None:
        """Connect to a checked, already-running TCP backend.

        The server is deliberately not owned by this client: ``shutdown`` only stops
        a locally spawned process. The scans and model/export locations passed in
        subsequent calls are paths on the server's filesystem.
        """
        host = host.strip()
        if not host:
            raise ValueError("Remote backend host cannot be empty.")
        if not 1 <= int(port) <= 65535:
            raise ValueError("Remote backend port must be between 1 and 65535.")
        with self._lock:
            if self._process is not None or self._host is not None:
                raise RuntimeError("Backend already started or connected.")
        # leave no half-configured client behind if the endpoint is not a LongiTrack server
        key_path = Path(private_key).expanduser() if private_key else None
        self._healthcheck(host, int(port), key_path)
        with self._lock:
            if self._process is not None or self._host is not None:
                raise RuntimeError("Backend already started or connected.")
            self._host = (host, int(port))
            self._tcp_private_key = key_path
        try:
            self._start_session()
        except Exception:
            with self._lock:
                self._host = None
                self._tcp_private_key = None
            raise

    @staticmethod
    def _open_tcp(host: str, port: int, timeout: float = 5.0):
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # The timeout above protects only initial connection establishment.
        sock.settimeout(None)
        return sock

    @staticmethod
    def _authenticate_tcp_socket(sock, private_key: Path | None) -> dict:
        protocol.send_json(sock, protocol.make_request("health", uuid.uuid4().hex))
        response = protocol.recv_json(sock)
        if response.get("type") != "result" or response.get("protocol") != 1:
            raise BackendError("Endpoint did not identify itself as a compatible LongiTrack backend.")
        if not response.get("authentication_required"):
            return response
        if response.get("authentication") != "ed25519" or private_key is None:
            raise BackendError("Remote backend requires an Ed25519 private key.")
        key = serialization.load_pem_private_key(private_key.read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise BackendError("Remote private key must be an Ed25519 PEM key.")
        raw_public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        key_id = hashlib.sha256(raw_public).hexdigest()
        challenge = base64.b64decode(response["challenge"], validate=True)
        signature = base64.b64encode(key.sign(challenge)).decode("ascii")
        protocol.send_json(
            sock,
            protocol.make_request("authenticate", uuid.uuid4().hex, key_id=key_id, signature=signature),
        )
        authenticated = protocol.recv_json(sock)
        if authenticated.get("type") != "result" or not authenticated.get("authenticated"):
            raise BackendError(authenticated.get("message", "remote authentication failed"))
        return response

    @classmethod
    def _healthcheck(
        cls, host: str, port: int, private_key: Path | None = None, timeout: float = 5.0,
    ) -> dict:
        sock = cls._open_tcp(host, port, timeout)
        try:
            return cls._authenticate_tcp_socket(sock, private_key)
        finally:
            BackendClient._close_socket(sock)

    @classmethod
    def probe_tcp(cls, host: str, port: int, **kwargs) -> dict:
        """Check a remote endpoint without changing this client's connection mode."""
        return cls._healthcheck(host.strip(), int(port), **kwargs)

    @property
    def is_remote(self) -> bool:
        with self._lock:
            return self._host is not None

    def _wait_ready(self, timeout: float = 60.0) -> None:
        assert self._process is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            code = self._process.poll()
            if code is not None:
                raise BackendError(f"Backend process exited during start-up (code {code}).")
            try:
                probe = self._connect()
            except OSError:
                time.sleep(0.05)
            else:
                probe.close()
                return
        raise BackendError("Backend process did not become ready in time.")

    def is_running(self) -> bool:
        with self._lock:
            return self._host is not None or (self._process is not None and self._process.poll() is None)

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
            remote = self._host is not None
        if process is None:
            if remote:
                try:
                    self._request("close_session", {"session_id": self._session_id}, progress=None)
                except Exception:
                    pass
                with self._lock:
                    self._host = None
                    self._tcp_private_key = None
                    self._session_id = None
            return
        try:
            self._request("shutdown", {}, progress=None)
        except Exception:  # noqa: BLE001 - best-effort; kill() below is the real guarantee
            pass
        self._terminate()

    def health(self) -> dict:
        """Return the connected backend's lightweight protocol identity."""
        return self._request("health", {}, progress=None)

    def _terminate(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            socket_path, self._socket_path = self._socket_path, None
            self._tcp_private_key = None
            self._session_id = None
            active = list(self._active_sockets)
            self._active_sockets.clear()
        for sock in active:
            self._close_socket(sock)
        if process is None or process.poll() is not None:
            if socket_path:
                self._unlink_socket(socket_path)
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()  # the backend never spawns children of its own, so this is final
        if socket_path:
            self._unlink_socket(socket_path)

    @staticmethod
    def _unlink_socket(path: str) -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def _discard_dead_process_locked(self) -> None:
        if self._process is None or self._process.poll() is None:
            return
        self._process = None
        if self._socket_path:
            self._unlink_socket(self._socket_path)
        self._socket_path = None

    # ------------------------------------------------------------------ wire --
    def _connect(self) -> socket.socket:
        if self._socket_path is not None:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self._socket_path)
        elif self._host is not None:
            sock = self._open_tcp(*self._host)
            self._authenticate_tcp_socket(sock, self._tcp_private_key)
        else:
            raise RuntimeError("Backend not started: call ensure_started() or connect_tcp() first.")
        return sock

    def _request(self, request_type: str, fields: dict, progress: Progress | None) -> dict:
        if request_type not in {"health", "open_session", "close_session", "shutdown"}:
            if self._session_id is None:
                raise RuntimeError("Backend session has not been opened.")
            fields = {**fields, "session_id": self._session_id}
        job_id = uuid.uuid4().hex
        sock = self._connect()
        with self._lock:
            self._active_sockets.add(sock)
        try:
            protocol.send_json(sock, protocol.make_request(request_type, job_id, **fields))
            while True:
                message = protocol.recv_json(sock)
                kind = message.get("type")
                if kind == "progress":
                    if progress is not None:
                        progress(message["message"])
                    continue
                if kind == "error":
                    raise BackendError(message.get("message", "backend request failed"))
                if kind == "result":
                    return protocol.recv_result(sock, message)
                raise protocol.ProtocolError(f"unexpected message type {kind!r}")
        finally:
            with self._lock:
                self._active_sockets.discard(sock)
            self._close_socket(sock)

    def _start_session(self) -> None:
        self._session_id = uuid.uuid4().hex
        self._request("open_session", {"session_id": self._session_id}, progress=None)

    @staticmethod
    def _close_socket(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def cancel_active(self) -> None:
        """Disconnect active requests so the GUI worker stops waiting immediately.

        CUDA kernels are not safely interruptible from another thread. The backend
        finishes its current serialized job, while the client drops its result and later
        requests wait in the backend queue. See kill() for actually freeing the backend
        up instead of just no longer waiting on it.
        """
        with self._lock:
            active = list(self._active_sockets)
        for sock in active:
            self._close_socket(sock)

    def has_active_request(self) -> bool:
        """Whether a request is genuinely in flight against the backend right now."""
        with self._lock:
            return bool(self._active_sockets)

    def kill(self) -> None:
        """Force-stop the backend process immediately, abandoning any request in flight.

        Unlike shutdown(), this never asks the backend to exit gracefully first -- that
        request would just queue up behind whatever CUDA call is already running on the
        backend's single connection at a time, taking exactly as long to answer as
        waiting the original call out, defeating the point of a cancel that is meant to
        be immediate. The next ensure_started() call spawns a fresh process; the model
        (and every per-scan cache) is gone and needs reinitializing.
        """
        self._terminate()

    # --------------------------------------------------------------- requests --
    def initialize(self, location: str | None, device: str | None, progress: Progress | None = None) -> dict:
        return self._request("initialize", {"location": location, "device": device}, progress)

    def load_scan(self, path: str, progress: Progress | None = None) -> dict:
        return self._request("load_scan", {"path": str(path)}, progress)

    def load_scans(self, paths: list[str], progress: Progress | None = None) -> list[dict]:
        return self._request("load_scans", {"paths": [str(path) for path in paths]}, progress)["scans"]

    @staticmethod
    def _scan_digest(path: Path) -> tuple[str, int, str]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        suffixes = path.suffixes
        suffix = "".join(suffixes[-2:]) if suffixes[-2:] == [".nii", ".gz"] else path.suffix
        if not suffix:
            raise ValueError(f"Cannot upload {path}: the file has no suffix.")
        return digest.hexdigest(), size, suffix

    def upload_file(self, path: str | Path, progress: Progress | None = None) -> str:
        """Copy one local file once into content-addressed backend CPU storage."""
        source_path = Path(path).expanduser()
        for attempt in range(2):
            try:
                return self._upload_file_once(source_path, progress)
            except BackendError as error:
                if attempt or "checksum does not match" not in str(error):
                    raise
                if progress is not None:
                    progress(f"{source_path.name} changed during upload; retrying once...")
        raise AssertionError("unreachable")

    def _upload_file_once(self, source_path: Path, progress: Progress | None) -> str:
        digest, size, suffix = self._scan_digest(source_path)
        if self._session_id is None:
            raise RuntimeError("Backend session has not been opened.")
        metadata = {"sha256": digest, "size": size, "suffix": suffix, "session_id": self._session_id}
        available = self._request("scan_exists", metadata, progress=None)
        if available["available"]:
            return str(available["path"])
        if progress is not None:
            progress(f"Uploading {source_path.name} to the backend ({size / 1_024 / 1_024:.0f} MiB)...")
        sock = self._connect()
        with self._lock:
            self._active_sockets.add(sock)
        try:
            job_id = uuid.uuid4().hex
            protocol.send_json(sock, protocol.make_request("upload_scan", job_id, **metadata))
            with source_path.open("rb") as source:
                protocol.send_file(sock, source, size)
            message = protocol.recv_json(sock)
            if message.get("type") == "error":
                raise BackendError(message.get("message", "scan upload failed"))
            if message.get("type") != "result":
                raise protocol.ProtocolError(f"unexpected upload response type {message.get('type')!r}")
            result = protocol.recv_result(sock, message)
            if progress is not None:
                progress(f"Uploaded {source_path.name} to the backend.")
            return str(result["path"])
        finally:
            with self._lock:
                self._active_sockets.discard(sock)
            self._close_socket(sock)

    def upload_scan(self, path: str | Path, progress: Progress | None = None) -> str:
        """Copy one local scan once into content-addressed backend CPU storage."""
        return self.upload_file(path, progress)

    def upload_model_folder(self, folder: str | Path, progress: Progress | None = None) -> str:
        """Transfer a local LongiSeg checkpoint and materialize it on the backend."""
        from longitrack_backend.model import CHECKPOINT_NAME, REQUIRED_FILES, validate_model_folder

        source = validate_model_folder(folder, folds=None)
        files = [source / name for name in REQUIRED_FILES]
        files.extend(sorted(source.glob(f"fold_*/{CHECKPOINT_NAME}")))
        if progress is not None:
            progress(f"Preparing local model {source.name} for backend upload...")
        entries = []
        for file_path in files:
            if not file_path.is_file():
                raise FileNotFoundError(f"Required model file is missing: {file_path}")
            entries.append(
                {"relative": str(file_path.relative_to(source)), "path": self.upload_file(file_path, progress)}
            )
        result = self._request("materialize_model", {"files": entries}, progress)
        if progress is not None:
            progress(f"Local model ready on backend at {result['folder']}.")
        return str(result["folder"])

    def preload_registration_scans(self, paths: list[str], progress: Progress | None = None) -> None:
        self._request("preload_registration_scans", {"paths": [str(path) for path in paths]}, progress)

    def warm_up_registration(self, progress: Progress | None = None) -> dict:
        """Load and exercise the fixed UniGradICON model before scans are needed."""
        return self._request("warm_up_registration", {}, progress)

    def release_scan(self, path: str) -> None:
        self._request("release_scan", {"path": str(path)}, None)

    def propagate(
        self,
        baseline_path: str,
        followup_path: str,
        points: list[list[float]],
        followup_shape: tuple[int, ...],
        refinement_steps: int | None = None,
        fast: bool | None = None,
        progress: Progress | None = None,
    ) -> list:
        # a plain dataclass, so importing it pulls in nothing heavy
        from longitrack_backend.registration import PointPropagation

        # Retain the old keyword for external callers while the UI uses the explicit
        # five-position refinement scale.
        if fast is not None:
            refinement_steps = None if fast else 50
        result = self._request(
            "propagate",
            {
                "baseline_path": str(baseline_path),
                "followup_path": str(followup_path),
                "points": [[float(c) for c in p] for p in points],
                "followup_shape": list(followup_shape),
                "refinement_steps": refinement_steps,
            },
            progress,
        )
        return [
            PointPropagation(
                baseline_index=p["baseline_index"],
                followup_index=p["followup_index"],
                out_of_bounds=p["out_of_bounds"],
                error=p["error"],
            )
            for p in result["propagations"]
        ]

    def invalidate_registration(
        self, clear_tracking: bool = False, clear_scans: bool = False
    ) -> None:
        self._request(
            "invalidate_registration",
            {
                "clear_tracking": bool(clear_tracking),
                "clear_scans": bool(clear_scans),
            },
            None,
        )

    def track(
        self,
        baseline_path: str,
        baseline_point: list[float],
        followup_path: str,
        followup_point: list[float],
        segment_baseline: bool = True,
        lesion_id: str | None = None,
        lesion_number: int | None = None,
        propagated_point: list[float] | None = None,
        disable_tta: bool = False,
        progress: Progress | None = None,
    ) -> dict:
        return self._request(
            "track",
            {
                "baseline_path": str(baseline_path),
                "baseline_point": [float(c) for c in baseline_point],
                "followup_path": str(followup_path),
                "followup_point": [float(c) for c in followup_point],
                "segment_baseline": bool(segment_baseline),
                "lesion_id": lesion_id,
                "lesion_number": lesion_number,
                "propagated_point": [float(c) for c in propagated_point] if propagated_point is not None else None,
                "disable_tta": bool(disable_tta),
            },
            progress,
        )

    def export(self, folder: str, lesion_ids: list[str], patient: str = "case_000",
               timepoints: list[str] | None = None, progress: Progress | None = None) -> list[str]:
        result = self._request(
            "export", {"folder": str(folder), "lesion_ids": list(lesion_ids), "patient": patient,
                       "timepoints": list(timepoints or ("baseline", "followup"))}, progress
        )
        return result["written"]
