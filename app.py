import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response
import threading

app = Flask(__name__)

# Initialize MediaPipe for segmentation
#Importing selfie_segmentation module
mp_selfie_segmentation = mp.solutions.selfie_segmentation
#create an instance of the SelfieSegmentation class, model_selection=1 argument specifies which segmentation model to use.
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Load and preprocess the background image
background = cv2.imread('background.jpeg')
if background is None:
    raise FileNotFoundError("Error: Background image 'background.jpeg' not found. Ensure the file path is correct.")

# Initialize video capture
cap = cv2.VideoCapture(0)
frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
background_resized = cv2.resize(background, (frame_width, frame_height))

def generate_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get segmentation mask
        results = segmentation.process(frame_rgb)
        mask = results.segmentation_mask

        # Refine the mask (threshold and smooth)
        _, refined_mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)
        refined_mask = cv2.GaussianBlur(refined_mask, (31, 31), 0)

        # Prepare masks
        foreground_mask = refined_mask.astype(np.float32)
        background_mask = 1.0 - foreground_mask

        # Extract the foreground and background
        foreground = cv2.multiply(frame.astype(np.float32), foreground_mask[..., None])
        background_segmented = cv2.multiply(background_resized.astype(np.float32), background_mask[..., None])

        # Combine the two
        output_frame = cv2.add(foreground, background_segmented)
        output_frame = output_frame.astype(np.uint8)  # Convert back to uint8

        # Convert frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpeg', output_frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
