from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

results = model.train("../dataset/dataset.yaml", epochs=10, batch_size=8, imgsz=640, workers=4)