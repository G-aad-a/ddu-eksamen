
import time
from gpiozero import DistanceSensor
from picamera import PiCamera
import numpy as np
from PIL import Image, ImageDraw
from pycoral.adapters import detect, common
from pycoral.utils.edgetpu import make_interpreter
from io import BytesIO

# Initialize ultrasonic sensor
echo_pin = 17
trigger_pin = 4
#ultrasonic = DistanceSensor(echo=echo_pin, trigger=trigger_pin)

def estimated_distance(area):
    return (area / 4e7) ** (-1 / 2.21)

# Model initialization (verify model is Edge TPU compiled)
model_path = 'model.tflite'
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Get input details (verify model requirements)
input_details = interpreter.get_input_details()
print(input_details[0]['dtype'])  # Expected: `np.float32`
print(input_details[0]['shape'])
input_shape = input_details[0]['shape']
_, height, width, _ = input_shape  # Should match model's expected input

# Initialize camera with model-compatible resolution
picam2 = PiCamera()
picam2.resolution = (width, height)  # Match model input resolution



while True:
    print("Capturing image...")
    stream = BytesIO()
    picam2.capture(stream, format='jpeg')
    stream.seek(0)

    # Process image for model input
    image = Image.open(stream).convert('RGB').resize((width, height), Image.NEAREST)
    _, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

    # Remove normalization (quantized models use uint8)
    #input_data = np.expand_dims(np.asarray(image), axis=0)
    
    # Preprocess

    input_data = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension -> [1, 640, 640, 3]
    # Run inference

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output using correct method
    output_details = interpreter.get_output_details()
    if len(output_details) == 1:  # Single output tensor (modern models)
        output = interpreter.tensor(output_details[0]['index'])()
        detections = detect.get_objects(
            interpreter,
            score_threshold=0.5, 
            image_scale=scale
        )
    else:  # Legacy 4-output format
        detections = detect.get_objects(interpreter, score_threshold=0.5)

    # Rest of your processing...
    draw = ImageDraw.Draw(image)
    for detection in detections:
        bbox = detection.bbox  # Bounding box coordinates: [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        
        width = xmax - xmin
        height = ymax - ymin
        area = width * height
        
        est_distance = estimated_distance(area)
        
        print(f"Detected object at ({x_center:.2f}, {y_center:.2f}), Size: {area:.2f}, Estimated Distance: {est_distance:.2f} cm")
        
        # Draw bounding box and label on the image
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="green", width=3)
        label_text = f"{est_distance:.2f}cm (Est) (Act)"
        draw.text((xmin, ymin - 10), label_text, fill="green")
    
    filename = "annotated_image.jpg"
    image.save(filename)
    time.sleep(5)
