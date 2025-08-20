from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="/Users/apple/Desktop/ithinka2/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu"
)


