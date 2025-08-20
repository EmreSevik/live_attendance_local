from ultralytics import YOLO
import cv2
import face_recognition
import pickle
import numpy as np
from datetime import datetime
import csv
import time

with open("face_db.pickle", "rb") as f:
    known_face_encodings, known_face_names, known_face_ids = pickle.load(f)

attendance_file = 'attendance.csv'
kayitli_olanlar = set()

try:
    with open(attendance_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            kayitli_olanlar.add(row[0])
except FileNotFoundError:
    pass

model = YOLO("runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
prev_time = time.time()
fps = 0

last_box = None
last_box_time = 0
BOX_TIMEOUT = 0.7

while True:
    ret, frame = cap.read()
    frame_count += 1
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    fps_text = f"FPS: {fps:.1f}"
    cv2.rectangle(frame, (10, 10), (150, 45), (130, 0, 75), -1)
    cv2.putText(frame, fps_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

    found_face = False

    if frame_count % 1 == 0:
        results = model.predict(frame, imgsz=640, conf=0.5)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        for i, box in enumerate(boxes):
            if int(classes[i]) != 0:
                continue
            x1, y1, x2, y2 = [int(c) for c in box]
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.shape[0] < 50 or person_crop.shape[1] < 50:
                continue
            try:
                person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(person_crop_rgb)
                if len(face_locations) == 0:
                    continue
                face_encodings = face_recognition.face_encodings(person_crop_rgb, face_locations)
                for encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_face_encodings, encoding)
                    if len(face_distances) > 0 and matches[np.argmin(face_distances)]:
                        idx = np.argmin(face_distances)
                        face_id = known_face_ids[idx]
                        name = known_face_names[idx]
                        label = f"{face_id} - {name}"

                        if face_id not in kayitli_olanlar:
                            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            with open(attendance_file, 'a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([face_id, name, now, "", ""])
                            print(f"{name} ({face_id}) saved for enrtrance: {now}")
                            kayitli_olanlar.add(face_id)

                        last_box = (x1, y1, x2, y2, label)
                        last_box_time = time.time()
                        found_face = True
            except Exception as e:
                continue


    if last_box and (time.time() - last_box_time < BOX_TIMEOUT):
        x1, y1, x2, y2, label = last_box
        kutu_renk = (0,255,0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), kutu_renk, 3)
        cv2.rectangle(frame, (x1, y1-30), (x2, y1), kutu_renk, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        last_box = None

    cv2.imshow("Entrance Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
