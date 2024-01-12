from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

try:
    results = model.train(data="dataset/dataset.yaml", epochs=200, imgsz=640)

except Exception as e:
    print(e)