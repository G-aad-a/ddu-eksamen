import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter
from picamera import PiCamera
from io import BytesIO

# Load TFLite model and allocate tensors.
interpreter = Interpreter("model.tflite")
interpreter.allocate_tensors()

# Get input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details[0]['dtype'])  # Expected: `np.float32`
print(input_details[0]['shape'])
input_shape = input_details[0]['shape']
_, height, width, _ = input_shape  # Should match model's expected input

# Initialize camera with model-compatible resolution
picam2 = PiCamera()
picam2.resolution = (width, height)  # Match model input resolution


stream = BytesIO()
picam2.capture(stream, format='jpeg')
stream.seek(0)

# Process image for model input
image = Image.open(stream).convert('RGB').resize((width, height), Image.NEAREST)
input_data = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

# Set input tensor.
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference.
interpreter.invoke()

# Get output tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])  # Shape: [1, 5, 8400]
predictions = np.squeeze(output_data).T  # Transpose to [8400, 5]

# Extract bounding boxes, confidence scores, and class IDs.
scores = predictions[:, 4]  # Confidence scores
boxes = predictions[:, :4]  # Bounding boxes [x_center, y_center, width, height]
class_ids = np.argmax(predictions[:, 4:], axis=1)  # Class IDs

# Filter detections by confidence threshold (e.g., >= 0.5).
confidence_threshold = 0.5
mask = scores >= confidence_threshold
boxes = boxes[mask]
scores = scores[mask]
class_ids = class_ids[mask]

# Convert bounding boxes from normalized coordinates to pixel coordinates.
image_width, image_height = image.size
boxes[:, [0, 2]] *= image_width   # x_center and width
boxes[:, [1, 3]] *= image_height  # y_center and height

# Print detections.
for i in range(len(scores)):
    print(f"Class ID: {class_ids[i]}, Confidence: {scores[i]:.2f}, Box: {boxes[i]}")
