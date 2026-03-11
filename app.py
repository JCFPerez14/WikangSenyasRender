import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from v8.real_time_asl_recognition_new import ASLSignRecognizer
import logging

app = Flask(__name__)
# Enable CORS for all routes (allows your frontend to connect)
CORS(app)

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Initialize the recognizer (model and mediapipe setup)
try:
    recognizer = ASLSignRecognizer()
    logging.info("ASL Sign Recognizer initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize ASL Sign Recognizer: {e}")
    recognizer = None

@app.route('/api/recognize_frame', methods=['POST'])
def recognize_frame():
    """
    Endpoint that accepts base64 encoded images from the client,
    processes them, and returns the ASL prediction.
    """
    if not recognizer:
        return jsonify({"error": "Model failed to initialize properly"}), 500

    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    image_data = data['image']
    
    # Check for the data URI header 'data:image/jpeg;base64,' and strip it if present
    header_str = "base64,"
    if header_str in image_data:
        image_data = image_data.split(header_str)[1]

    try:
        # Decode base64 to numpy array
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        # Decode image using OpenCV
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
             return jsonify({"error": "Failed to decode image"}), 400

        # Process frame
        result = recognizer.process_frame(frame)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Basic endpoint to verify the API is up and running."""
    status = "healthy" if recognizer else "model_failed"
    return jsonify({"status": status})

if __name__ == '__main__':
    # It is recommended to run this script using your virtual environment explicitly
    # Example: .\venv\Scripts\python.exe app.py
    app.run(host='0.0.0.0', port=5000, debug=True)
