import base64
import requests
import cv2
import numpy as np
import time

# Create a dummy image (black square)
img = np.zeros((480, 640, 3), dtype=np.uint8)

# Encode as JPEG
_, buffer = cv2.imencode('.jpg', img)

# Convert to base64 string
b64_str = base64.b64encode(buffer).decode('utf-8')
data_uri = f"data:image/jpeg;base64,{b64_str}"

url = "http://127.0.0.1:5000/api/recognize_frame"

try:
    print("Testing /api/health...")
    health_res = requests.get("http://127.0.0.1:5000/api/health")
    print(f"Health Response: {health_res.json()}")

    print("\nSending frame to /api/recognize_frame...")
    response = requests.post(url, json={"image": data_uri})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

except Exception as e:
    print(f"Error: {e}")
