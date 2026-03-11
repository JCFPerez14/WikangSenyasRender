import sys
print(sys.path)
try:
    import mediapipe as mp
    print(f"MediaPipe file: {mp.__file__}")
    print(f"Solutions: {mp.solutions}")
except Exception as e:
    print(f"Error: {e}")
