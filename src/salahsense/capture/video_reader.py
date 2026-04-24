"""Video frame reader for phase-1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


@dataclass(frozen=True)
class FramePacket:
    """A single decoded frame plus metadata."""

    frame_index: int
    timestamp_ms: int
    frame_bgr: cv2.typing.MatLike


class VideoReader:
    """Read frames from a local video file with optional downscaling."""

    def __init__(self, video_path: str, process_width: int) -> None:
        self.video_path = Path(video_path)
        self.process_width = process_width

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")

        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._fps = fps if fps > 1e-6 else 30.0

    def frames(self) -> Iterator[FramePacket]:
        """Yield resized frames with monotonic timestamps for MediaPipe VIDEO mode."""
        frame_index = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break

            frame_index += 1
            frame = self._resize_if_needed(frame)
            timestamp_ms = int((frame_index / self._fps) * 1000)
            yield FramePacket(frame_index=frame_index, timestamp_ms=timestamp_ms, frame_bgr=frame)

    def close(self) -> None:
        self._cap.release()

    def _resize_if_needed(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        height, width = frame.shape[:2]
        if width <= self.process_width:
            return frame

        scale = self.process_width / float(width)
        new_size = (self.process_width, int(height * scale))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
