import cv2
import mediapipe as mp
import numpy as np
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import settings
except ImportError:
    pass

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetector:
    """
    Wrapper for MediaPipe Face Landmarker (New Tasks API).
    """
    def __init__(self, static_image_mode=False, max_num_faces=1, 
                 refine_landmarks=True, min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        
        if not hasattr(settings, 'FACE_LANDMARKER_PATH') or not os.path.exists(settings.FACE_LANDMARKER_PATH):
             raise FileNotFoundError(f"Model file not found at {getattr(settings, 'FACE_LANDMARKER_PATH', 'Unknown')}. Please download face_landmarker.task.")

        # Create FaceLandmarker options
        base_options = python.BaseOptions(model_asset_path=settings.FACE_LANDMARKER_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process(self, image):
        """
        Processes an image and returns the face landmarks.
        Args:
            image: A BGR image from OpenCV.
        Returns:
            results: FaceLandmarkerResult object.
        """
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect
        detection_result = self.detector.detect(mp_image)
        
        return detection_result

    def draw_landmarks(self, image, results):
        """
        Draws the face mesh landmarks on the image using manual drawing (Points only).
        We avoid mp.solutions.drawing_utils as mp.solutions is not available in this version.
        Args:
            image: BGR image.
            results: FaceLandmarkerResult object.
        """
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                h, w, _ = image.shape
                for landmark in face_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    # Draw small circle
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
                    
    def get_landmarks_as_np(self, results, image_width, image_height):
        """
        Converts landmarks to a numpy array of (x, y) coordinates.
        """
        if not results.face_landmarks:
            return None
            
        # Assume single face
        face_landmarks = results.face_landmarks[0]
        landmarks_np = np.array([
            (int(point.x * image_width), int(point.y * image_height)) 
            for point in face_landmarks
        ])
        
        return landmarks_np
