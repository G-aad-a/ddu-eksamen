from flask import Flask, render_template_string, Response
from PIL import Image
import io
import threading
import time

app = Flask(__name__)

latest_image = None
lock = threading.Lock()

HTML = """
<!doctype html>
<title>Live Object Feed</title>
<h1>Live Feed</h1>
<img src="{{ url_for('video_feed') }}">
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    while True:
        with lock:
            if latest_image is not None:
                buffer = io.BytesIO()
                latest_image.save(buffer, format='JPEG')
                frame = buffer.getvalue()
            else:
                continue  

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.03)



def run_flask(): # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def set_image(image: Image.Image):
    global latest_image
    with lock: # Bruger lock for at sikre tråd-sikkerhed
        latest_image = image.copy()
