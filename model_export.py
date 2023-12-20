from ultralytics import YOLO

# https://docs.ultralytics.com/modes/export/#key-features-of-export-mode

#  "Performance: Gain up to 5x GPU speedup with TensorRT and 3x CPU speedup with ONNX or OpenVINO."

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom trained model

# Export the model
model.export(format='onnx')