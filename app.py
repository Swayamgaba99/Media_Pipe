import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response
import threading

app = Flask(__name__)

# Initialize MediaPipe for segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Load the background image
background = cv2.imread('background.jpeg')
if background is None:
    print("Error: Background image not loaded properly.")

# Initialize video capture (using webcam or you can replace this with a stream URL)
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Read frame from the video capture
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get segmentation mask
        results = segmentation.process(frame_rgb)
        mask = results.segmentation_mask

        # Apply Gaussian blur to smooth edges of the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Resize background to match the frame dimensions
        background_resized = cv2.resize(background, (frame.shape[1], frame.shape[0]))

        # Invert the mask to apply it correctly
        background_mask = np.ones_like(mask) - mask

        # Apply the mask to extract the foreground (person)
        foreground = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8))

        # Apply the mask to extract the background
        background_segmented = cv2.bitwise_and(background_resized, background_resized, mask=background_mask.astype(np.uint8))

        # Combine the foreground and background
        output_frame = cv2.add(foreground, background_segmented)

        # Convert frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpeg', output_frame)
        if not ret:
            continue

        # Yield the frame in the format required by Flask to send it to the browser
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

# Run Flask server in a separate thread to avoid blocking the main thread
if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
