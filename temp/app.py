import cv2
import numpy as np
import json
import os
import pandas as pd
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime
import logging
import time

logging.basicConfig(filename="face_recognition.log", level=logging.ERROR)

def load_roi(filename="roi_config_first_row.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return np.array(data["roi"], np.int32)
    except FileNotFoundError:
        print("ROI file not found")
        return None

polygon_roi = load_roi()
model = YOLO("yolov11n-face.pt")

KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

attendance_file = "attendance.csv"
attendance_today = {}

# Load today's attendance
today_date = datetime.now().strftime("%Y-%m-%d")
if os.path.exists(attendance_file):
    df = pd.read_csv(attendance_file)
    attendance_today = {
        name: True for name in df[df["Date"] == today_date]["Name"]
    }

cap = cv2.VideoCapture("data/NVR_ch23_main_20250424113539_20250424114529.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
target_interval = 0.4
frame_skip = int(fps * target_interval)

cropped_faces_display = {}

def mark_attendance(name, face_crop):
    global attendance_today, cropped_faces_display
    today_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    if name not in attendance_today:
        attendance_today[name] = True
        new_entry = pd.DataFrame({"Name": [name], "Date": [today_date], "Time": [current_time]})
        new_entry.to_csv(attendance_file, mode="a", header=False, index=False)
        print(f"Attendance Marked for {name}")

        if name not in cropped_faces_display:
            cropped_faces_display[name] = {
                "image": face_crop,
                "time": time.time()
            }
        return True
    return False

def is_inside_polygon(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

face_embeddings_cache = {}
processed_faces = set()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [polygon_roi], 255)
    roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

    results = model(roi_frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_center_x = (x1 + x2) // 2
            face_center_y = (y1 + y2) // 2

            if is_inside_polygon(face_center_x, face_center_y, polygon_roi):
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                try:
                    DeepFace.analyze(face_crop, actions=["emotion"], enforce_detection=True)

                    verification = DeepFace.find(
                        img_path=face_crop,
                        db_path=KNOWN_FACES_DIR,
                        model_name="ArcFace",
                        enforce_detection=False
                    )

                    if isinstance(verification, list) and len(verification) > 0 and not verification[0].empty:
                        best_match = verification[0].iloc[0]
                        person_name = os.path.basename(best_match["identity"]).split(".")[0]

                        if person_name not in face_embeddings_cache:
                            face_embeddings_cache[person_name] = True

                            if person_name not in processed_faces:
                                if mark_attendance(person_name, face_crop):
                                    processed_faces.add(person_name)

                        cv2.putText(frame, f"{person_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                except Exception as e:
                    logging.error(f"Face Verification Error: {str(e)}")
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.polylines(frame, [polygon_roi], isClosed=True, color=(255, 0, 0), thickness=2)

    # Display recently recognized faces without overlap
    keys_to_remove = []
    for idx, (person_name, face_info) in enumerate(cropped_faces_display.items()):
        cropped_face_image = face_info["image"]
        display_time = face_info["time"]

        if time.time() - display_time > 2:
            keys_to_remove.append(person_name)
            continue

        height, width = cropped_face_image.shape[:2]
        max_size = 100
        scale_factor = max_size / max(height, width)
        resized_face = cv2.resize(cropped_face_image, (int(width * scale_factor), int(height * scale_factor)))

        corner_x = frame.shape[1] - resized_face.shape[1] - 10
        corner_y = 10 + idx * (resized_face.shape[0] + 40)

        frame[corner_y:corner_y + resized_face.shape[0], corner_x:corner_x + resized_face.shape[1]] = resized_face
        cv2.putText(frame, "Present", (corner_x, corner_y + resized_face.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    for key in keys_to_remove:
        del cropped_faces_display[key]

    cv2.imshow("Face Recognition Attendance", frame)

    key = cv2.waitKey(int(1000 / fps))
    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
