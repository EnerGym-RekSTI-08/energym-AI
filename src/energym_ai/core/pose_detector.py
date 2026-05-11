from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float

    def as_xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    def as_xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class PoseResult:
    landmarks: list[Landmark] | None
    raw_results: any = None

    @property
    def detected(self) -> bool:
        return self.landmarks is not None

    def get(self, idx: int) -> Landmark:
        if self.landmarks is None:
            raise RuntimeError("Tidak ada pose terdeteksi pada frame ini.")
        return self.landmarks[idx]


_POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
])

_MODEL_NAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}

_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def _ensure_model(model_complexity: int) -> str:
    model_dir = Path.cwd() / "models"
    model_dir.mkdir(exist_ok=True)
    path = model_dir / _MODEL_NAMES[model_complexity]
    if not path.exists():
        from loguru import logger
        url = _MODEL_URLS[model_complexity]
        logger.info(f"Downloading MediaPipe model: {url}")
        urllib.request.urlretrieve(url, path)
        logger.info(f"Model saved → {path}")
    return str(path)


class PoseDetector:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True,
    ) -> None:
        model_path = _ensure_model(model_complexity)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._detector = mp_vision.PoseLandmarker.create_from_options(options)
        self._last_ts: int = 0

    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        if ts_ms <= self._last_ts:
            ts_ms = self._last_ts + 1
        self._last_ts = ts_ms
        result = self._detector.detect_for_video(mp_image, ts_ms)

        if not result.pose_landmarks:
            return PoseResult(landmarks=None, raw_results=result)

        landmarks = [
            Landmark(lm.x, lm.y, lm.z, lm.visibility)
            for lm in result.pose_landmarks[0]
        ]
        return PoseResult(landmarks=landmarks, raw_results=result)

    def draw(self, frame_bgr: np.ndarray, result: PoseResult) -> np.ndarray:
        if not result.detected:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        lms = result.landmarks
        for lm in lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (cx, cy), 4, (0, 255, 0), -1)
        for a, b in _POSE_CONNECTIONS:
            if a < len(lms) and b < len(lms):
                ax, ay = int(lms[a].x * w), int(lms[a].y * h)
                bx, by = int(lms[b].x * w), int(lms[b].y * h)
                cv2.line(frame_bgr, (ax, ay), (bx, by), (0, 255, 255), 2)
        return frame_bgr

    def close(self) -> None:
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
