import os
import time
import base64
from functools import lru_cache

import numpy as np
import cv2
from flask import Flask, request, jsonify
import mediapipe as mp

app = Flask(__name__)

# -------- Utilities --------

def now_ts() -> int:
    return int(time.time())

def decode_b64_image(b64: str):
    """
    Accepts:
      - raw base64
      - data URL: "data:image/jpeg;base64,...."
    Returns BGR image (OpenCV) or None
    """
    try:
        if not b64:
            return None
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64, validate=False)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def brightness_score(img_bgr) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def resize_for_speed(img_bgr, target_width: int = 320):
    """
    Reduce CPU/RAM by resizing while keeping aspect ratio.
    """
    h, w = img_bgr.shape[:2]
    if w <= target_width:
        return img_bgr
    scale = target_width / float(w)
    new_w = target_width
    new_h = max(1, int(h * scale))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Lazy initialization (prevents heavy boot / reduces crash risk)
@lru_cache(maxsize=1)
def get_face_detector():
    return mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6
    )

def analyze_frame(img_bgr):
    """
    Returns a dict: face_count, brightness, risk, verdict, events
    """
    events = []
    risk = 0

    # Speed up
    img_bgr = resize_for_speed(img_bgr, target_width=320)
    h, w = img_bgr.shape[:2]

    # 1) Darkness / camera covered
    bright = brightness_score(img_bgr)
    if bright < 35:
        events.append({"type": "DARK_FRAME", "score": round(35 - bright, 2)})
        risk += 2

    # 2) Face detection
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detector = get_face_detector()
    res = detector.process(rgb)
    faces = res.detections or []
    face_count = len(faces)

    if face_count == 0:
        events.append({"type": "NO_FACE", "score": 1})
        risk += 3
    elif face_count > 1:
        events.append({"type": "MULTIPLE_FACES", "score": face_count})
        risk += 5

    # 3) Simple off-center heuristic (MVP)
    if face_count == 1:
        det = faces[0]
        bbox = det.location_data.relative_bounding_box

        cx = (bbox.xmin + bbox.width / 2.0) * w
        dx = abs(cx - (w / 2.0)) / (w / 2.0)  # 0..1

        if dx > 0.55:
            events.append({"type": "LOOKING_AWAY_OR_OFF_CENTER", "score": round(dx, 2)})
            risk += 2

        # face too small => student far away
        if bbox.width < 0.18:
            events.append({"type": "FACE_TOO_SMALL", "score": round(bbox.width, 3)})
            risk += 1

    # Verdict
    verdict = "OK"
    if risk >= 6:
        verdict = "HIGH_RISK"
    elif risk >= 3:
        verdict = "WARNING"

    return {
        "face_count": int(face_count),
        "brightness": round(float(bright), 2),
        "risk": int(risk),
        "verdict": verdict,
        "events": events
    }

# -------- Routes --------

@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "service": "ai-proctoring-api",
        "ts": now_ts(),
        "endpoints": ["/health", "/analyze"]
    })

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "status": "running",
        "ts": now_ts()
    })

@app.post("/analyze")
def analyze():
    """
    JSON Body:
    {
      "frame": "data:image/jpeg;base64,...",
      "student_id": "S1001",
      "session_id": "sess_123"
    }
    """
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get("frame")
    student_id = data.get("student_id", "unknown")
    session_id = data.get("session_id", "unknown")

    img = decode_b64_image(frame_b64)
    if img is None:
        return jsonify({"ok": False, "error": "bad_or_missing_image"}), 400

    try:
        result = analyze_frame(img)
        return jsonify({
            "ok": True,
            "student_id": student_id,
            "session_id": session_id,
            "ts": now_ts(),
            **result
        })
    except Exception as e:
        # avoid crashing worker
        return jsonify({
            "ok": False,
            "error": "analyze_failed",
            "detail": str(e)[:200]
        }), 500

# -------- Local run (Railway uses gunicorn) --------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
