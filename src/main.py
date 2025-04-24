from PIL import ImageDraw
import threading
import time
from server import run_flask, set_image
from model import detect_objects, estimated_distance
from motor import move_forward, move_backward, move_left, move_right, rotate, shoot, stop, motor
from enum import Enum
from sys import exit


class State(Enum):
    SEARCHING = 0
    TARGETING = 1
    SHOOTING = 2
    IDLE = 3

current_state = State.SEARCHING

def drain_water():
    for i in range(1, 10):
       shoot()

if __name__ == '__main__':
    threading.Thread(target=run_flask, daemon=True).start()

    while True:
        boxes, scores, class_ids, image = detect_objects(conf_thresh=0.4, iou_thresh=0.5)

        draw = ImageDraw.Draw(image)
        for box, score, class_id in zip(boxes, scores, class_ids):
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

    
        set_image(image) # For flask siden!
        if len(boxes) == 0:
            print("No objects detected")
            current_state = State.SEARCHING

            #move_forward(512, 0.95)
            time.sleep(0.1)
            rotate(25)

            continue

        current_state = State.TARGETING
        for box, score, class_id in zip(boxes, scores, class_ids):
            x_min, y_min, x_max, y_max = box

            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            area = (x_max - x_min) * (y_max - y_min)
            est_distance = estimated_distance(area)

            x_min, y_min, x_max, y_max = boxes[0]
            object_center_x = (x_min + x_max) / 2

            pixel_offset = object_center_x - 320
            degrees_per_pixel = 62.2 / 640
            angle_offset = pixel_offset * degrees_per_pixel / 2

            print(f"Object is {pixel_offset:+.1f} px from center → turn {angle_offset:+.2f}° {+est_distance:.2f} cm away")
            if abs(angle_offset) > 1.75:
                rotate(angle_offset)
            else:
                if est_distance >= 200:
                    print("moving long forward")
                    move_forward(512, 2.5)
                else:
                    move_forward(512, 0.95)

            if 35 >= est_distance:
                print("skyd")
                current_state = State.SHOOTING
                shoot()
                
                
                exit(0)
                time.sleep(1.5)
                rotate(180)
                current_state = State.SEARCHING
                
                break

