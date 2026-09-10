"""
Phase 3 — FastAPI backend (scaffold)

Receives base64-encoded JPEG frames over a WebSocket from the frontend,
runs them through AUExtractor (Module 1), buffers a sliding window, and
(once you've trained it) feeds the window into MicroExpressionLSTM
(Module 2) to get live stress/confidence scores back to the dashboard.

Run:
    pip install fastapi uvicorn[standard] websockets python-multipart
    uvicorn backend.app.main:app --reload

This is a working skeleton: the AU extraction path is fully wired up,
the model-inference path is stubbed with TODOs until you complete Phase 2.
"""

import base64
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .au_extraction import AUExtractor

app = FastAPI(title="Micro-Expression & Cognitive Load Analyzer")

WINDOW_SIZE = 30  # frames (~1 second at 30 FPS)


def decode_base64_frame(b64_string: str) -> np.ndarray:
    """Frontend sends 'data:image/jpeg;base64,...' — strip the prefix if present."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.websocket("/ws/analyze")
async def analyze_stream(websocket: WebSocket):
    await websocket.accept()
    extractor = AUExtractor()
    feature_window = deque(maxlen=WINDOW_SIZE)

    # TODO (Phase 2 complete): load your trained model once here, e.g.
    #   model = MicroExpressionLSTM(...)
    #   model.load_state_dict(torch.load("checkpoints/best_model.pt"))
    #   model.eval()

    try:
        while True:
            payload = await websocket.receive_json()
            frame_b64 = payload.get("frame")
            if not frame_b64:
                continue

            rgb_frame = decode_base64_frame(frame_b64)
            features = extractor.process(rgb_frame)
            feature_window.append(features.as_vector())

            response = {
                "face_detected": features.face_detected,
                "ear_avg": features.ear_avg,
                "au4_brow_dist": features.au4_brow_dist,
                "au12_mouth_dist": features.au12_mouth_dist,
                "window_filled": len(feature_window) == WINDOW_SIZE,
            }

            # TODO (Phase 2 complete): once feature_window is full, stack it
            # into a tensor of shape (1, WINDOW_SIZE, num_features), run it
            # through `model`, and add the results to `response`, e.g.:
            #
            #   if len(feature_window) == WINDOW_SIZE:
            #       x = torch.tensor([list(feature_window)], dtype=torch.float32)
            #       logits, load_score = model(x)
            #       response["cognitive_load"] = load_score.item()
            #       response["expression_class"] = logits.argmax(dim=-1).item()

            await websocket.send_json(response)

    except WebSocketDisconnect:
        pass
    finally:
        extractor.close()
