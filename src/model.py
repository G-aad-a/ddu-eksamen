import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from picamera import PiCamera
from io import BytesIO

# --- Utility Functions ---

def estimated_distance(area):
    return (area / 4e7) ** (-1 / 2.21)

def iou(box1, box2):
    xa, ya = max(box1[0], box2[0]), max(box1[1], box2[1])
    xb, yb = min(box1[2], box2[2]), min(box1[3], box2[3])
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
        indices = [i for i in indices[1:] if iou(boxes[current], boxes[i]) < iou_thresh]
    return keep

# --- Detection Pipeline Functions ---

def load_model(model_path):
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter

picam = PiCamera()


def capture_image(width, height):
    picam.resolution = (width, height)
    stream = BytesIO()
    picam.capture(stream, format='jpeg')
    stream.seek(0)
    return Image.open(stream).convert("RGB")

def preprocess_image(image, input_details):
    input_dtype = input_details[0]['dtype']
    scale, zero_point = input_details[0]['quantization']
    image = image.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
    np_img = np.array(image).astype(np.float32)

    if input_dtype == np.float32:
        input_data = np_img / 255.0
    elif input_dtype == np.int8:
        input_data = (np_img / 255.0 - 0.0) / scale + zero_point
        input_data = np.clip(input_data, -128, 127).astype(np.int8)
    else:
        raise ValueError("Unsupported input dtype")

    return np.expand_dims(input_data, axis=0)

def run_inference(interpreter, input_data):
    input_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

def process_output(output_data, input_details, output_details, original_size, conf_thresh=0.1, iou_thresh=0.5):
    if output_data.dtype == np.int8:
        scale, zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - zero_point) * scale

    predictions = np.squeeze(output_data).T
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    class_ids = np.zeros(len(scores), dtype=int)

    mask = scores >= conf_thresh
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

    # Convert to pixel format
    ow, oh = original_size
    boxes[:, [0, 2]] *= ow
    boxes[:, [1, 3]] *= oh

    pixel_boxes = []
    for box in boxes:
        cx, cy, w, h = box
        x_min, y_min = cx - w / 2, cy - h / 2
        x_max, y_max = cx + w / 2, cy + h / 2
        pixel_boxes.append([x_min, y_min, x_max, y_max])
    pixel_boxes = np.array(pixel_boxes)

    keep = non_max_suppression(pixel_boxes, scores, iou_thresh=iou_thresh)
    return pixel_boxes[keep], scores[keep], class_ids[keep]



model_path="best_full_integer_quant_edgetpu.tflite"
interpreter = load_model(model_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]


def detect_objects(conf_thresh=0.1, iou_thresh=0.5):

    image = capture_image(width, height)
    original_size = image.size
    input_data = preprocess_image(image, input_details)
    output_data = run_inference(interpreter, input_data)
    boxes, scores, class_ids = process_output(
        output_data, input_details, output_details, original_size, conf_thresh, iou_thresh
    )

    # If we have detections, only keep the highest confidence one
    # Sort by confidence score (highest first)
    # Even though we already use non_max_suppression, we can still have multiple boxes with the same score
    # So we sort and take the first one

    if len(boxes) > 0:
        sorted_indices = np.argsort(scores)[::-1]  
        boxes = boxes[sorted_indices]
        scores = scores[sorted_indices]
        class_ids = class_ids[sorted_indices]

        boxes = boxes[:1]
        scores = scores[:1]
        class_ids = class_ids[:1]
    else:
        boxes = np.array([])
        scores = np.array([])
        class_ids = np.array([])


    #for box, score, cls in zip(boxes, scores, class_ids):
        #x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        #print(f"Object detected at ({x:.1f}, {y:.1f}), Score: {score:.2f}, Class ID: {cls}")

    return boxes, scores, class_ids, image  
