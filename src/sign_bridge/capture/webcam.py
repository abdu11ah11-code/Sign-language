"""Threaded webcam capture.

A background thread continuously pulls frames so the consumer (inference
loop / UI) always reads the *latest* frame without blocking on camera I/O.
Tracks a rolling FPS estimate and survives transient camera disconnects.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from types import TracebackType

import cv2

from sign_bridge.logging_conf import get_logger
from sign_bridge.typing import NDArray

log = get_logger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or recovers no frames."""


class ThreadedCamera:
    """Continuously grabs frames from an OpenCV device on a daemon thread.

    Args:
        index: OS camera index.
        width: Requested capture width (driver may adjust).
        height: Requested capture height.
        target_fps: Requested capture FPS.
        mirror: Horizontally flip frames (selfie view).
        reconnect_attempts: Consecutive read failures tolerated before the
            thread tries to reopen the device; after a second full round of
            failures the camera is marked dead and :meth:`read` raises.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 30,
        mirror: bool = True,
        reconnect_attempts: int = 30,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._target_fps = target_fps
        self._mirror = mirror
        self._reconnect_attempts = reconnect_attempts

        self._cap: cv2.VideoCapture | None = None
        self._frame: NDArray | None = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._dead = threading.Event()
        self._frame_times: deque[float] = deque(maxlen=60)
        self._thread: threading.Thread | None = None

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> ThreadedCamera:
        """Open the device and start the grab thread.

        Returns:
            self, for chaining / ``with ThreadedCamera(...).start() as cam``.

        Raises:
            CameraError: If the device cannot be opened at all.
        """
        self._cap = self._open()
        self._thread = threading.Thread(target=self._loop, name="camera-grab", daemon=True)
        self._thread.start()
        log.info("camera_started", index=self._index, width=self._width, height=self._height)
        return self

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            raise CameraError(
                f"Cannot open camera index {self._index}. "
                "Check that it exists and no other app is using it."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        return cap

    def _loop(self) -> None:
        failures = 0
        reopened_once = False
        while not self._stopped.is_set():
            assert self._cap is not None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= self._reconnect_attempts:
                    if reopened_once:
                        log.error("camera_dead", index=self._index)
                        self._dead.set()
                        return
                    log.warning("camera_reconnecting", index=self._index)
                    self._cap.release()
                    try:
                        self._cap = self._open()
                        reopened_once = True
                        failures = 0
                    except CameraError:
                        self._dead.set()
                        return
                time.sleep(0.02)
                continue

            failures = 0
            if self._mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame
            self._frame_times.append(time.monotonic())

    # ── consumption ──────────────────────────────────────────────────────
    def read(self) -> NDArray:
        """Return a copy of the most recent frame.

        Blocks briefly until the first frame arrives.

        Raises:
            CameraError: If the device died and could not be recovered.
        """
        deadline = time.monotonic() + 5.0
        while True:
            if self._dead.is_set():
                raise CameraError("Camera disconnected and could not be recovered.")
            with self._lock:
                if self._frame is not None:
                    return self._frame.copy()
            if time.monotonic() > deadline:
                raise CameraError("Camera produced no frames within 5 s.")
            time.sleep(0.01)

    @property
    def fps(self) -> float:
        """Rolling FPS estimate over the last ~60 frames (0.0 until warm)."""
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    @property
    def is_alive(self) -> bool:
        """False once the camera is marked unrecoverable."""
        return not self._dead.is_set()

    def release(self) -> None:
        """Stop the grab thread and release the device."""
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        log.info("camera_released", index=self._index)

    def __enter__(self) -> ThreadedCamera:
        return self if self._thread is not None else self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
