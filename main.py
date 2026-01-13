import os, time, base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
import mediapipe as mp

app = Flask(__name__)

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

def decode_b64_image(b64: str):
    # b64 can be "data:image/jpeg;base64,...."
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    img_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img

def brightness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

@app.post("/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get("frame")
    student_id = data.get("student_id")
    session_id = data.get("session_id")

    if not frame_b64:
        return jsonify({"ok": False, "error": "missing frame"}), 400

    img = decode_b64_image(frame_b64)
    if img is None:
        return jsonify({"ok": False, "error": "bad image"}), 400

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1) Darkness check
    bright = brightness_score(img)
    events = []
    risk = 0

    if bright < 35:  # tweak later
        events.append({"type": "DARK_FRAME", "score": round(35 - bright, 2)})
        risk += 2

    # 2) Face count check
    res = mp_face.process(rgb)
    faces = res.detections or []
    face_count = len(faces)

    if face_count == 0:
        events.append({"type": "NO_FACE", "score": 1})
        risk += 3
    elif face_count > 1:
        events.append({"type": "MULTIPLE_FACES", "score": face_count})
        risk += 5

    # 3) Simple "looking away" (MVP heuristic)
    # Use first face bbox position: if face center is too far from center => likely looking away / moving away
    if face_count == 1:
        det = faces[0]
        bbox = det.location_data.relative_bounding_box
        cx = (bbox.xmin + bbox.width / 2.0) * w
        cy = (bbox.ymin + bbox.height / 2.0) * h
        dx = abs(cx - (w / 2)) / (w / 2)  # 0..1
        if dx > 0.55:
            events.append({"type": "LOOKING_AWAY_OR_OFF_CENTER", "score": round(dx, 2)})
            risk += 2

    verdict = "OK"
    if risk >= 6:
        verdict = "HIGH_RISK"
    elif risk >= 3:
        verdict = "WARNING"

    return jsonify({
        "ok": True,
        "student_id": student_id,
        "session_id": session_id,
        "face_count": face_count,
        "brightness": round(bright, 2),
        "risk": risk,
        "verdict": verdict,
        "events": events,
        "ts": int(time.time())
    })
