
import sys
import os
import cv2
import numpy as np

# Add updated path
sys.path.append(os.getcwd())

try:
    print("Attempting to import utils.face_detection...")
    from utils.face_detection import FaceDetector
    print("Successfully imported FaceDetector.")
    
    print("Attempting to initialize FaceDetector...")
    detector = FaceDetector()
    print("Successfully initialized FaceDetector.")
    
    print("Running dummy inference...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.process(img)
    print("Processed image successfully.")
    
    print("Running draw_landmarks...")
    detector.draw_landmarks(img, results)
    print("Draw landmarks ran successfully.")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
