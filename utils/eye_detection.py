import numpy as np
import cv2
from scipy.spatial import distance as dist

class EyeDetector:
    def __init__(self):
        # MediaPipe Face Mesh Landmark Indices
        # Left Eye (Upper: 160, 158; Lower: 153, 144; Corners: 33, 133)
        self.RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
        # Right Eye (Upper: 385, 387; Lower: 373, 380; Corners: 362, 263)
        self.LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

    def get_eye_landmarks(self, face_landmarks):
        """
        Extracts eye landmarks from the full face landmarks.
        Args:
            face_landmarks: List of (x, y) tuples.
        Returns:
            left_eye, right_eye: Lists of (x, y) coordinates.
        """
        left_eye = np.array([face_landmarks[i] for i in self.LEFT_EYE_IDXS])
        right_eye = np.array([face_landmarks[i] for i in self.RIGHT_EYE_IDXS])
        return left_eye, right_eye

    def calculate_ear(self, eye):
        """
        Calculates the Eye Aspect Ratio (EAR).
        Args:
            eye: List of 6 (x, y) landmark coordinates.
        Returns:
            ear: The computed EAR value.
        """
        # Vertical distances
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        
        # Horizontal distance
        C = dist.euclidean(eye[0], eye[3])
        
        if C == 0:
            return 0.0

        ear = (A + B) / (2.0 * C)
        return ear

    def extract_eye_region(self, image, eye_landmarks, target_size=(224, 224)):
        """
        Crops and resizes the eye region for model inference.
        Args:
            image: Full frame image.
            eye_landmarks: (x, y) coordinates of the eye.
            target_size: Tuple (width, height) for resizing.
        Returns:
            processed_eye: Preprocessed image ready for model inference.
        """
        # Get bounding box
        x_min = np.min(eye_landmarks[:, 0])
        x_max = np.max(eye_landmarks[:, 0])
        y_min = np.min(eye_landmarks[:, 1])
        y_max = np.max(eye_landmarks[:, 1])
        
        # Add padding
        margin = 10
        x_min = max(0, x_min - margin)
        x_max = min(image.shape[1], x_max + margin)
        y_min = max(0, y_min - margin)
        y_max = min(image.shape[1], y_max + margin) # Note: squared usually better?
        
        # Crop
        eye_img = image[y_min:y_max, x_min:x_max]
        
        if eye_img.size == 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        # Resize
        try:
            eye_resized = cv2.resize(eye_img, target_size)
        except Exception:
             return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        return eye_resized
