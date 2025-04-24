import numpy as np
from PIL import Image, ImageDraw
from pycoral.utils.edgetpu import make_interpreter
from picamera import PiCamera
from io import BytesIO

def estimated_distance(area):
    return (area / 4e7) ** (-1 / 2.21)

def iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

def non_max_suppression(boxes, scores, iou_thresh=0.5):
    indices = scores.argsort()[::-1]
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        rest = indices[1:]
        indices = [i for i in rest if iou(boxes[current], boxes[i]) < iou_thresh]
    return keep

# Load model
MODEL_PATH = "best_float32.tflite"
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

# Input/output info
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
_, height, width, _ = input_shape

print("Model dtype:", input_details[0]['dtype'])
print("Input shape:", input_shape)
print("Input quantization:", input_details[0]['quantization'])
print("Output quantization:", output_details[0]['quantization'])

# Capture image
picam = PiCamera()
picam.resolution = (width, height)
stream = BytesIO()
picam.capture(stream, format='jpeg')
stream.seek(0)

# Preprocess image
image = Image.open(stream).convert("RGB")
original_image = image.copy()
image = image.resize((width, height))

input_dtype = input_details[0]['dtype']
scale, zero_point = input_details[0]['quantization']
if input_dtype == np.float32:
    input_data = np.array(image).astype(np.float32) / 255.0
elif input_dtype == np.int8:
    input_data = np.array(image).astype(np.float32)
    input_data = (input_data / 255.0 - 0.0) / scale + zero_point
    input_data = np.clip(input_data, -128, 127).astype(np.int8)
else:
    raise ValueError("Unsupported input dtype")

input_data = np.expand_dims(input_data, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get and dequantize output
output_data = interpreter.get_tensor(output_details[0]['index'])
scale, zero_point = output_details[0]['quantization']
if output_data.dtype == np.int8:
    output_data = (output_data.astype(np.float32) - zero_point) * scale

# Parse predictions
predictions = np.squeeze(output_data).T
boxes = predictions[:, :4]
scores = predictions[:, 4]
class_ids = np.zeros(len(scores), dtype=int)  # assume one class

# Filter
confidence_threshold = 0.1
mask = scores >= confidence_threshold
boxes = boxes[mask]
scores = scores[mask]
class_ids = class_ids[mask]

# Convert boxes to pixel coordinates
original_width, original_height = original_image.size
boxes[:, [0, 2]] *= original_width   # x_center, width
boxes[:, [1, 3]] *= original_height  # y_center, height

# Convert to [x_min, y_min, x_max, y_max] format
converted_boxes = []
for box in boxes:
    x_center, y_center, w, h = box
    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2
    converted_boxes.append([x_min, y_min, x_max, y_max])
converted_boxes = np.array(converted_boxes)

# Apply NMS
keep_indices = non_max_suppression(converted_boxes, scores, iou_thresh=0.5)
boxes = converted_boxes[keep_indices]
scores = scores[keep_indices]
class_ids = class_ids[keep_indices]

# Draw detections
draw = ImageDraw.Draw(original_image)
for box, score, class_id in zip(boxes, scores, class_ids):
    x_min, y_min, x_max, y_max = box
    print(f"Detected object at ({(x_min + x_max) / 2:.2f}, {(y_min + y_max) / 2:.2f})")
    print(f"Class ID: {class_id}, Confidence: {score:.2f}, Box: {box}")

    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
    label = f"Class {class_id}: {score:.2f}"
    text_w, text_h = draw.textsize(label)
    draw.rectangle([x_min, y_min - text_h - 4, x_min + text_w, y_min], fill="red")
    draw.text((x_min, y_min - text_h - 4), label, fill="white")

# Save result
original_image.save("image_result2.jpg")
print("Saved result as image_result.jpg")
