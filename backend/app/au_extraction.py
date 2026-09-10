"""
Module 1 — Landmark & Action Unit (AU) Extraction (reusable version)

Same logic as phase1_blink_and_au_poc.py, refactored into a class so the
FastAPI backend (Phase 3) can call `extractor.process(frame)` per incoming
WebSocket frame and get back a clean feature vector, instead of duplicating
the standalone script's logic.
"""

from dataclasses import dataclass

import mediapipe as mp
import numpy as np

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_BROW_INNER = 65
RIGHT_BROW_INNER = 295
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
LEFT_CHEEK = 234
RIGHT_CHEEK = 454


@dataclass
class FrameFeatures:
    ear_avg: float
    au4_brow_dist: float
    au12_mouth_dist: float
    face_detected: bool

    def as_vector(self):
        """Feature vector shape expected by MicroExpressionLSTM."""
        return [self.ear_avg, self.au4_brow_dist, self.au12_mouth_dist]


class AUExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def _euclidean(p1, p2):
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _point(self, landmarks, idx, w, h):
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)

    def _ear(self, landmarks, eye_idx, w, h):
        pts = [self._point(landmarks, i, w, h) for i in eye_idx]
        p1, p2, p3, p4, p5, p6 = pts
        vertical = self._euclidean(p2, p6) + self._euclidean(p3, p5)
        horizontal = self._euclidean(p1, p4)
        return vertical / (2.0 * horizontal) if horizontal else 0.0

    def process(self, rgb_frame: np.ndarray) -> FrameFeatures:
        """
        rgb_frame: HxWx3 numpy array, RGB order (convert with
                   cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) before calling this).
        """
        h, w = rgb_frame.shape[:2]
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return FrameFeatures(0.0, 0.0, 0.0, face_detected=False)

        landmarks = results.multi_face_landmarks[0].landmark

        ear_left = self._ear(landmarks, LEFT_EYE, w, h)
        ear_right = self._ear(landmarks, RIGHT_EYE, w, h)
        ear_avg = (ear_left + ear_right) / 2.0

        face_scale = self._euclidean(
            self._point(landmarks, LEFT_CHEEK, w, h),
            self._point(landmarks, RIGHT_CHEEK, w, h),
        )
        face_scale = face_scale or 1.0

        au4 = self._euclidean(
            self._point(landmarks, LEFT_BROW_INNER, w, h),
            self._point(landmarks, RIGHT_BROW_INNER, w, h),
        ) / face_scale

        au12 = self._euclidean(
            self._point(landmarks, LEFT_MOUTH_CORNER, w, h),
            self._point(landmarks, RIGHT_MOUTH_CORNER, w, h),
        ) / face_scale

        return FrameFeatures(ear_avg, au4, au12, face_detected=True)

    def close(self):
        self._face_mesh.close()
