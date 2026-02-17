
try:
    from mediapipe.tasks.python.vision import FaceLandmarker
    print("SUCCESS: mediapipe.tasks.python.vision.FaceLandmarker found.")
except ImportError as e:
    print(f"FAILURE: {e}")
