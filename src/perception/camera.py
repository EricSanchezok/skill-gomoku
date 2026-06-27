"""Camera abstraction for the Gomoku perception pipeline.

The RealSense device is used as a normal UVC webcam in this project. We only
need RGB frames for board-state recognition, so the production camera path uses
OpenCV ``VideoCapture`` instead of the RealSense depth SDK.
"""

from __future__ import annotations

import logging
from types import TracebackType

import cv2
import numpy as np

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


class WebCamera:
    """OpenCV-backed webcam camera.

    This class treats the connected camera as a standard UVC webcam and returns
    BGR images ready for OpenCV processing. It intentionally does not expose
    depth frames; ``get_frames()`` returns ``(color_bgr, None)`` for interface
    consistency with test cameras.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        warmup_frames: int = 5,
    ) -> None:
        """Open the webcam.

        Args:
            device_index: OpenCV camera index, usually ``0`` for the first camera.
            width: Requested frame width in pixels.
            height: Requested frame height in pixels.
            fps: Requested capture frame rate.
            warmup_frames: Number of initial frames to discard after opening.

        Raises:
            RuntimeError: If the camera cannot be opened or no frame can be read.
        """
        self._device_index = int(device_index)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._capture = cv2.VideoCapture(self._device_index)

        if not self._capture.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self._device_index}. "
                "Check the USB connection and whether another app is using the camera."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._capture.set(cv2.CAP_PROP_FPS, self._fps)

        last_frame: np.ndarray | None = None
        for _ in range(max(1, warmup_frames)):
            ok, frame = self._capture.read()
            if ok and frame is not None:
                last_frame = frame

        if last_frame is None:
            self.release()
            raise RuntimeError(
                f"Camera index {self._device_index} opened but returned no frames. "
                "Try a different camera.device_index or reconnect the camera."
            )

        actual_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        logger.info(
            "WebCamera opened: index=%d requested=%dx%d@%d actual=%dx%d@%.1f",
            self._device_index,
            self._width,
            self._height,
            self._fps,
            actual_w,
            actual_h,
            actual_fps,
        )

    def get_frames(self) -> tuple[np.ndarray, None]:
        """Capture one color frame.

        Returns:
            ``(color_bgr, None)``. The second value is always ``None`` because
            this project uses the camera as a normal webcam and does not read
            depth data.

        Raises:
            RuntimeError: If frame capture fails.
        """
        ok, color_bgr = self._capture.read()
        if not ok or color_bgr is None:
            raise RuntimeError(
                f"Failed to read frame from camera index {self._device_index}. "
                "The camera may have been disconnected or claimed by another app."
            )
        return color_bgr, None

    def get_color(self) -> np.ndarray:
        """Capture and return a BGR ``uint8`` color frame."""
        color_bgr, _ = self.get_frames()
        return color_bgr

    def get_intrinsics(self) -> dict[str, float | int | None]:
        """Return approximate camera metadata.

        Standard webcams do not expose calibrated intrinsics through OpenCV, so
        focal length and principal point are returned as ``None``. The actual
        capture resolution is included for downstream code that needs it.
        """
        return {
            "fx": None,
            "fy": None,
            "ppx": None,
            "ppy": None,
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

    def release(self) -> None:
        """Release the OpenCV capture handle."""
        if hasattr(self, "_capture") and self._capture.isOpened():
            self._capture.release()
            logger.info("WebCamera released: index=%d", self._device_index)

    def __enter__(self) -> WebCamera:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()


class MockCamera:
    """Synthetic camera that generates a stable checkerboard frame for tests."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        enable_depth: bool = False,
    ) -> None:
        """Create a mock camera.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Ignored; kept for config compatibility.
            enable_depth: When true, ``get_frames`` returns a synthetic depth
                image. Production webcam mode ignores depth.
        """
        self._width = int(width)
        self._height = int(height)
        self._enable_depth = bool(enable_depth)
        self._cached_frame: np.ndarray | None = None

        logger.info("MockCamera created: %dx%d (no hardware)", self._width, self._height)

    def get_frames(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Return a synthetic color frame and optional synthetic depth."""
        color_bgr = self._generate_checkerboard()

        if self._enable_depth:
            depth_mm = np.linspace(300, 2000, self._width, dtype=np.uint16)
            depth_mm = np.tile(depth_mm, (self._height, 1))
            return color_bgr, depth_mm

        return color_bgr, None

    def get_color(self) -> np.ndarray:
        """Return the synthetic color frame (BGR ``uint8``)."""
        return self._generate_checkerboard()

    def get_intrinsics(self) -> dict[str, float | int]:
        """Return simple synthetic intrinsics for tests."""
        fx = self._width / 2.0
        fy = fx
        return {
            "fx": fx,
            "fy": fy,
            "ppx": self._width / 2.0,
            "ppy": self._height / 2.0,
            "width": self._width,
            "height": self._height,
        }

    def release(self) -> None:
        """No-op; mock camera has no hardware resources to release."""
        logger.info("MockCamera released.")

    def __enter__(self) -> MockCamera:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()

    def _generate_checkerboard(self) -> np.ndarray:
        """Generate a color checkerboard frame, cached after first call."""
        if self._cached_frame is not None:
            return self._cached_frame

        square = 80
        rows = self._height // square + 1
        cols = self._width // square + 1

        tile = np.zeros((rows, cols), dtype=np.uint8)
        tile[1::2, ::2] = 1
        tile[::2, 1::2] = 1

        mask = np.repeat(np.repeat(tile, square, axis=0), square, axis=1)
        mask = mask[: self._height, : self._width]

        dark = np.array([40, 40, 140], dtype=np.uint8)
        light = np.array([200, 200, 240], dtype=np.uint8)
        frame = np.where(mask[..., None], light, dark).astype(np.uint8)
        self._cached_frame = frame
        return frame


def create_camera(config: dict | None = None, mock: bool = False) -> WebCamera | MockCamera:
    """Create the configured camera.

    Args:
        config: Full project config dictionary. When omitted, ``config/default.yaml``
            is loaded.
        mock: When true, return ``MockCamera`` regardless of hardware.

    Returns:
        ``WebCamera`` for real hardware or ``MockCamera`` for tests.
    """
    if config is None:
        config = load_config()

    cam_cfg = config.get("camera", {})
    width = int(cam_cfg.get("width", 1280))
    height = int(cam_cfg.get("height", 720))
    fps = int(cam_cfg.get("fps", 30))

    if mock:
        return MockCamera(
            width=width,
            height=height,
            fps=fps,
            enable_depth=bool(cam_cfg.get("enable_depth", False)),
        )

    return WebCamera(
        device_index=int(cam_cfg.get("device_index", 0)),
        width=width,
        height=height,
        fps=fps,
        warmup_frames=int(cam_cfg.get("warmup_frames", 5)),
    )
