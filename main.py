import os
import time
import base64
from functools import lru_cache

import numpy as np
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)

# -------------------------
# Utils
# -------------------------

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
    h, w = img_bgr.shape[:2]
    if w <= target_width:
        return img_bgr
    scale = target_width / float(w)
    new_w = target_width
    new_h = max(1, int(h * scale))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

# -------------------------
# Face Detection Engines
#   1) MediaPipe (if available)
#   2) OpenCV Haar Cascade fallback
# -------------------------

MP_AVAILABLE = False
MP_ERROR = None

try:
    import mediapipe as mp  # type: ignore
    # Some deployments may have a broken mediapipe install; check "solutions"
    if hasattr(mp, "solutions"):
        MP_AVAILABLE = True
    else:
        MP_ERROR = "mediapipe imported but has no attribute 'solutions'"
except Exception as e:
    MP_ERROR = f"mediapipe import failed: {e}"

@lru_cache(maxsize=1)
def get_mp_face_detector():
    """
    Lazy init MediaPipe detector (only if MP_AVAILABLE=True)
    """
    if not MP_AVAILABLE:
        return None
    return mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6
    )

@lru_cache(maxsize=1)
def get_haar_detector():
    """
    Haar cascade fallback (very lightweight)
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)

def detect_faces(img_bgr):
    """
    Returns face_count and a list of face boxes.
    Each box is (x, y, w, h) in pixels.
    """
    img_bgr = resize_for_speed(img_bgr, 320)
    h, w = img_bgr.shape[:2]

    # 1) Try MediaPipe first
    if MP_AVAILABLE:
        detector = get_mp_face_detector()
        if detector is not None:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            res = detector.process(rgb)
            detections = res.detections or []
            boxes = []
            for det in detections:
                bbox = det.location_data.relative_bounding_box
                x = int(max(0, bbox.xmin) * w)
                y = int(max(0, bbox.ymin) * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                boxes.append((x, y, bw, bh))
            return len(boxes), boxes, "mediapipe"

    # 2) Fallback: Haar cascade
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    haar = get_haar_detector()
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    boxes = [(int(x), int(y), int(bw), int(bh)) for (x, y, bw, bh) in faces]
    return len(boxes), boxes, "opencv_haar"

def analyze_frame(img_bgr):
    events = []
    risk = 0

    img_bgr = resize_for_speed(img_bgr, 320)
    h, w = img_bgr.shape[:2]

    # Darkness check
    bright = brightness_score(img_bgr)
    if bright < 35:
        events.append({"type": "DARK_FRAME", "score": round(35 - bright, 2)})
        risk += 2

    # Face detection
    face_count, boxes, engine = detect_faces(img_bgr)

    if face_count == 0:
        events.append({"type": "NO_FACE", "score": 1})
        risk += 3
    elif face_count > 1:
        events.append({"type": "MULTIPLE_FACES", "score": face_count})
        risk += 5

    # Simple off-center heuristic using first face
    if face_count == 1:
        x, y, bw, bh = boxes[0]
        cx = x + bw / 2.0
        dx = abs(cx - (w / 2.0)) / (w / 2.0)

        if dx > 0.55:
            events.append({"type": "LOOKING_AWAY_OR_OFF_CENTER", "score": round(dx, 2)})
            risk += 2

        # face too small => far
        rel_w = (bw / float(w)) if w else 0.0
        if rel_w < 0.18:
            events.append({"type": "FACE_TOO_SMALL", "score": round(rel_w, 3)})
            risk += 1

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
        "events": events,
        "engine": engine
    }

# -------------------------
# Routes
# -------------------------

@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "service": "ai-proctoring-api",
        "ts": now_ts(),
        "endpoints": ["/health", "/debug", "/analyze"]
    })

@app.get("/health")
def health():
    return jsonify({"ok": True, "status": "running", "ts": now_ts()})

@app.get("/debug")
def debug():
    info = {
        "ok": True,
        "ts": now_ts(),
        "mp_available": MP_AVAILABLE,
        "mp_error": MP_ERROR,
    }

    # Extra info if mediapipe loaded
    try:
        if MP_AVAILABLE:
            import mediapipe as mp2  # re-import
            info["mp_version"] = getattr(mp2, "__version__", "unknown")
            info["has_solutions"] = hasattr(mp2, "solutions")
            info["mp_file"] = getattr(mp2, "__file__", "unknown")
    except Exception as e:
        info["mp_debug_error"] = str(e)

    return jsonify(info)

@app.post("/analyze")
def analyze():
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
        return jsonify({
            "ok": False,
            "error": "analyze_failed",
            "detail": str(e)[:220]
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
