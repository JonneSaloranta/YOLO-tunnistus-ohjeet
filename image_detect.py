from ultralytics import YOLO

model = YOLO("./best.pt")  # oma malli

results = model("./test_img/cat_6.jpeg")  # pass the frame to the model