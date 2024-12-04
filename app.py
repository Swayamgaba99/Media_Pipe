import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response
import threading

app = Flask(__name__)

# Initialize MediaPipe for segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Load and preprocess the background image
background = cv2.imread('background.jpeg')
if background is None:
    raise FileNotFoundError("Error: Background image 'background.jpeg' not found. Ensure the file path is correct.")

# Initialize video capture
cap = cv2.VideoCapture(0)

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """Adjust brightness and contrast of an image."""
    beta = brightness  # Brightness
    alpha = 1 + (contrast / 100.0)  # Contrast scale factor
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def generate_frames():
    global cap
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the background to match the frame dimensions
        frame_height, frame_width = frame.shape[:2]
        background_resized = cv2.resize(background, (frame_width, frame_height))

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get segmentation mask
        results = segmentation.process(frame_rgb)
        mask = results.segmentation_mask

        # Refine the mask (threshold and smooth)
        _, refined_mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)
        refined_mask = cv2.GaussianBlur(refined_mask, (15, 15), 0)  # Reduced kernel size

        # Prepare masks
        foreground_mask = refined_mask[..., None]  # Expand dimensions
        background_mask = 1.0 - foreground_mask  # Inverse mask for the background

        # Ensure masks have the same type and dimensions
        foreground_mask = np.repeat(foreground_mask, 3, axis=2).astype(np.float32) * 0.85  # Further dim foreground
        background_mask = np.repeat(background_mask, 3, axis=2).astype(np.float32)

        # Convert frame and background to float32 for element-wise operations
        frame_float = frame.astype(np.float32)
        background_resized = background_resized.astype(np.float32)

        # Extract foreground and background
        foreground = cv2.multiply(frame_float, foreground_mask)
        background_segmented = cv2.multiply(background_resized, background_mask)

        # Combine the two
        output_frame = cv2.add(foreground, background_segmented)
        output_frame = output_frame.astype(np.uint8)  # Convert back to uint8

        # Apply brightness and contrast adjustment
        output_frame = adjust_brightness_contrast(output_frame, brightness=-20, contrast=15)  # Increased brightness reduction

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
    try:
        threading.Thread(target=run_flask).start()
    except KeyboardInterrupt:
        pass
    finally:
        # Release resources
        cap.release()
        segmentation.close()
        print("Resources released.")
