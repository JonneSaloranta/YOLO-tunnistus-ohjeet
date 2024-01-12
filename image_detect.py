from ultralytics import YOLO

model = YOLO("./best.pt")  # oma malli tai jokin muu malli esim, yolov8n.pt

url = 'https://www.youtube.com/watch?v=LJvyb_WujZg'

results = model(url)  # pass the frame to the model