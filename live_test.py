from ultralytics import YOLO
import cv2

model = YOLO('/Users/apple/Desktop/ithinka2/runs/detect/train/weights/best.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model.predict(frame, imgsz=640, conf=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Webcam Live", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
