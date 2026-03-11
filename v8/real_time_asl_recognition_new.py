#!/usr/bin/env python3
"""
Real-time Sign Language Recognition using a pre-trained TFLite model.
Model: v8\asl_final_model_dynamic_quant.tflite
Label encoder: label_encoder.pkl
Features:
- 30‑frame sliding window buffer
- On‑screen buffering progress
- Live MediaPipe landmark overlay
"""

import cv2
import numpy as np
import mediapipe as mp
import pickle
import tensorflow as tf
from collections import deque
import os

# ---------------------------
# Constants (must match training)
# ---------------------------
SEQUENCE_LENGTH = 30          # number of frames per sample
N_FEATURES = 285              # landmark features per frame
TARGET_CLASSES = ['Ingat', 'Magandang Gabi', 'Magandang Hapon',
                  'Magandang Umaga', 'Mahal Kita', 'Paalam']


def extract_landmarks_from_frame(frame, holistic):
    """Process one frame and return a flattened 285‑element landmark array."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = holistic.process(frame_rgb)

    landmarks = []
    # Left hand (21 × 3 = 63)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    # Right hand (63)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    # Pose world landmarks (33 × 3 = 99)
    if results.pose_world_landmarks:
        for lm in results.pose_world_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 99)

    # Face (20 selected landmarks × 3 = 60)
    key_indices = [0, 1, 4, 5, 9, 10, 13, 14, 17, 18,
                   21, 33, 36, 39, 42, 45, 48, 51, 54, 57]
    if results.face_landmarks:
        for idx in key_indices:
            if idx < len(results.face_landmarks.landmark):
                lm = results.face_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0, 0.0, 0.0])
    else:
        landmarks.extend([0.0] * 60)

    return np.array(landmarks, dtype=np.float32)


class ASLSignRecognizer:
    def __init__(self, model_path='v8/asl_final_model_float32.tflite', encoder_path='v8/label_encoder.pkl'):
        # Resolve paths relative to the current working directory to support being run from different places
        if not os.path.exists(model_path) and os.path.exists(os.path.join(os.path.dirname(__file__), 'asl_final_model_float32.tflite')):
             model_path = os.path.join(os.path.dirname(__file__), 'asl_final_model_float32.tflite')
             encoder_path = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')

        # Load label encoder
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        self.classes = self.label_encoder.classes_
        print("Loaded classes:", self.classes)

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']      # expected [1, 30, 285]
        print(f"Model input shape: {self.input_shape}")

        # MediaPipe Holistic setup
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.last_stable_sign = None
        self.current_candidate = None
        self.current_sign_count = 0

    def process_frame(self, frame):
        """
        Process a single image frame, updating the buffer and running prediction if buffer is full.
        """
        # We assume `frame` is already a valid BGR image (OpenCV format)
        landmarks = extract_landmarks_from_frame(frame, self.holistic)
        self.frame_buffer.append(landmarks)

        # Buffer not full yet
        if len(self.frame_buffer) < SEQUENCE_LENGTH:
            return {
                "status": "buffering",
                "buffer_status": f"{len(self.frame_buffer)}/{SEQUENCE_LENGTH}",
                "sign": None,
                "confidence": 0.0
            }

        # Run inference
        input_data = np.array(self.frame_buffer, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, N_FEATURES)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        pred_index = np.argmax(output_data[0])
        pred_label = self.classes[pred_index]
        confidence = output_data[0][pred_index]

        stable_sign = None
        if confidence > 0.8:
            if pred_label == self.current_candidate:
                self.current_sign_count += 1
            else:
                self.current_candidate = pred_label
                self.current_sign_count = 1
                
            if self.current_sign_count == 3:
                stable_sign = pred_label
        else:
            self.current_candidate = None
            self.current_sign_count = 0

        return {
            "status": "success",
            "buffer_status": f"{SEQUENCE_LENGTH}/{SEQUENCE_LENGTH}",
            "sign": str(pred_label),
            "confidence": float(confidence),
            "stable_sign": stable_sign
        }

    def close(self):
        self.holistic.close()


def main():
    """Standalone mode using webcam"""
    recognizer = ASLSignRecognizer()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real‑time recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)          # mirror view
        
        # We can also process and render landmarks manually for visual feedback
        # but to test our class simply we'll call process_frame
        result = recognizer.process_frame(frame)
        
        # Display buffering progress while collecting initial frames
        if result["status"] == "buffering":
            progress_text = result["message"]
            cv2.putText(frame, progress_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        elif result["status"] == "success":
            text = f"{result['prediction']} ({result['confidence']:.2f})"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw MediaPipe landmarks for visual feedback
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = recognizer.holistic.process(frame_rgb)
        if results.pose_landmarks:
            recognizer.mp_drawing.draw_landmarks(frame, results.pose_landmarks, recognizer.mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            recognizer.mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, recognizer.mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            recognizer.mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, recognizer.mp_holistic.HAND_CONNECTIONS)
        if results.face_landmarks:
            recognizer.mp_drawing.draw_landmarks(frame, results.face_landmarks, recognizer.mp_holistic.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=recognizer.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1))

        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()

if __name__ == '__main__':
    main()