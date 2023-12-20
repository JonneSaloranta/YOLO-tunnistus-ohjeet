from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

try:
    results = model.train(data="dataset/dataset.yaml", epochs=10, batch=8, imgsz=640, workers=4)

except Exception as e:
    print(e)