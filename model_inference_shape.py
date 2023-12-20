import onnx
from onnx import shape_inference

# Load the ONNX model
model_path = './yolov8n.onnx'  # Update the model path
onnx_model = onnx.load(model_path)

# Perform shape inference to determine the input size
inferred_model = shape_inference.infer_shapes(onnx_model)

# Get the input information
input_info = inferred_model.graph.input[0]
input_name = input_info.name
input_shape = input_info.type.tensor_type.shape.dim

print("Input Name:", input_name)
print("Input Shape:", [dim.dim_value for dim in input_shape])
