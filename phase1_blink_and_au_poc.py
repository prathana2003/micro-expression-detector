"""
Phase 1 Proof of Concept
------------------------
Opens your webcam, tracks 468 3D face landmarks with MediaPipe Face Mesh,
computes:
  1. Eye Aspect Ratio (EAR) -> blink detection -> blink rate (blinks/min)
  2. Two Action Unit (AU) proxy distances:
       - AU4  (brow lowerer)   : distance between inner eyebrow points
       - AU12 (lip corner pull): distance between the two mouth corners

These raw signals are exactly what you will later buffer into a sliding
window and feed to the Phase 2 LSTM model as a time series.

Run:
    python phase1_blink_and_au_poc.py
Press 'q' to quit. Press 'l' to toggle CSV logging on/off.
"""

import csv
import time
from collections import deque

import cv2
import os
import numpy as np

# Use the MediaPipe Tasks API (FaceLandmarker) on newer mediapipe packages.
try:
    from mediapipe.tasks.python.vision import FaceLandmarker
    from mediapipe.tasks.python.vision import FaceLandmarkerOptions
    from mediapipe.tasks.python.vision import FaceLandmarksConnections
    from mediapipe.tasks.python.vision.core import image as mp_image
    from mediapipe.tasks.python.vision.core.image import ImageFormat
    USING_TASKS_API = True
except Exception:
    # Fall back to legacy solutions API if available (older mediapipe)
    import mediapipe as mp
    USING_TASKS_API = False

# ---------------------------------------------------------------------------
# Landmark index reference (MediaPipe Face Mesh, 468-point model)
# https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
# ---------------------------------------------------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]     # p1..p6 for EAR formula
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

LEFT_BROW_INNER = 65
RIGHT_BROW_INNER = 295
NOSE_BRIDGE = 6            # used to normalize brow distance by face scale

LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

EAR_BLINK_THRESHOLD = 0.21     # tune per-person; lower = stricter blink
EAR_CONSEC_FRAMES = 2          # frames EAR must stay below threshold to count as a blink


def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def eye_aspect_ratio(landmarks, eye_idx, w, h):
    """Standard EAR formula: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx]
    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def main(log_to_csv=False):
    # Path to the MediaPipe Face Landmarker model
    MODEL_PATH = "face_landmarker_v2.task"

    if USING_TASKS_API:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"MediaPipe Tasks model not found: {MODEL_PATH}.\n"
                "Download the face_landmarker_v2.task model and place it in the project root."
            )

        mp_drawing = __import__(
            'mediapipe.tasks.python.vision.drawing_utils',
            fromlist=['*']
        )

        mp_drawing_styles = __import__(
            'mediapipe.tasks.python.vision.drawing_styles',
            fromlist=['*']
        )

        # Open webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check camera permissions / index."
            )

        # Create the FaceLandmarker
        with FaceLandmarker.create_from_model_path(MODEL_PATH) as face_landmarker:
            _run_loop(
                face_landmarker,
                mp_drawing,
                mp_drawing_styles,
                log_to_csv,
                cap
            )

        return

    # If using old MediaPipe API
    raise RuntimeError("Failed to initialize MediaPipe Face Landmarker.")

def _run_loop(face_landmarker, mp_drawing, mp_drawing_styles, log_to_csv, cap):
    """Run the main capture loop using a FaceLandmarker Tasks instance."""
    blink_count = 0
    consec_below_thresh = 0
    session_start = time.time()

    feature_window = deque(maxlen=30)

    csv_file = None
    csv_writer = None
    if log_to_csv:
        csv_file = open("phase1_signals_log.csv", "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "ear_left", "ear_right", "au4_brow_dist",
                              "au12_mouth_dist", "blink_count"])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = mp_image.Image(ImageFormat.SRGB, rgb)
        results = face_landmarker.detect(mp_img)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]

            ear_left = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear_avg = (ear_left + ear_right) / 2.0

            if ear_avg < EAR_BLINK_THRESHOLD:
                consec_below_thresh += 1
            else:
                if consec_below_thresh >= EAR_CONSEC_FRAMES:
                    blink_count += 1
                consec_below_thresh = 0

            brow_l = get_point(landmarks, LEFT_BROW_INNER, w, h)
            brow_r = get_point(landmarks, RIGHT_BROW_INNER, w, h)
            face_scale = euclidean(get_point(landmarks, 234, w, h),
                                    get_point(landmarks, 454, w, h))
            au4_brow_dist = euclidean(brow_l, brow_r) / face_scale if face_scale else 0.0

            mouth_l = get_point(landmarks, LEFT_MOUTH_CORNER, w, h)
            mouth_r = get_point(landmarks, RIGHT_MOUTH_CORNER, w, h)
            au12_mouth_dist = euclidean(mouth_l, mouth_r) / face_scale if face_scale else 0.0

            feature_vector = [ear_avg, au4_brow_dist, au12_mouth_dist]
            feature_window.append(feature_vector)

            if csv_writer:
                csv_writer.writerow([time.time(), ear_left, ear_right,
                                      au4_brow_dist, au12_mouth_dist, blink_count])

            elapsed_min = max((time.time() - session_start) / 60.0, 1e-6)
            blink_rate = blink_count / elapsed_min

            overlay_lines = [
                f"Blinks: {blink_count}  (rate: {blink_rate:.1f}/min)",
                f"EAR: {ear_avg:.3f}",
                f"AU4 brow dist:  {au4_brow_dist:.3f}",
                f"AU12 mouth dist: {au12_mouth_dist:.3f}",
            ]
            for i, line in enumerate(overlay_lines):
                cv2.putText(frame, line, (10, 30 + 25 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Phase 1: Blink + AU Proxy PoC (press q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()
        print("Signals logged to phase1_signals_log.csv")


if __name__ == "__main__":
    main(log_to_csv=True)
