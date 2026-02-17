from scipy.spatial import distance as dist
import numpy as np

class YawnDetector:
    def __init__(self):
        # MediaPipe Face Mesh Landmark Indices for Lips
        # Inner lips generally used for MAR
        # Top: 13, Bottom: 14, Left: 78, Right: 308 (Outer lips used for robustness here)
        # Using a set of points to define the mouth opening
        self.MOUTH_IDXS = [
             [61, 291], # Corners (Left, Right) - Horizontal
             [39, 181], # Vertical 1 (Upper, Lower)
             [0, 17],   # Vertical 2 (Upper, Lower) - Center
             [269, 405] # Vertical 3 (Upper, Lower)
        ]
        # Indices in the flattened face_landmarks list
        # We need to map these correctly.
        # Let's use a standard MAR formula based on specific MediaPipe landmarks
        # Upper lip: 13, 81, 178, 311
        # Lower lip: 14, 178, 81, 311 
        # Simpler set:
        # P1: 78 (Left Corner), P2: 308 (Right Corner)
        # P3: 13 (Upper Center), P4: 14 (Lower Center)
        # P5: 82 (Upper Left), P6: 87 (Lower Left)
        # P7: 312 (Upper Right), P8: 317 (Lower Right)
        
        # Refined indices for MAR
        self.P1 = 78
        self.P2 = 308
        self.P3 = 13
        self.P4 = 14
        self.P5 = 82
        self.P6 = 87
        self.P7 = 312
        self.P8 = 317


    def calculate_mar(self, face_landmarks):
        """
        Calculates the Mouth Aspect Ratio (MAR).
        Args:
            face_landmarks: List of (x, y) tuples.
        Returns:
            mar: The computed MAR value.
        """
        # Vertical distances
        A = dist.euclidean(face_landmarks[self.P3], face_landmarks[self.P4]) # Center
        B = dist.euclidean(face_landmarks[self.P5], face_landmarks[self.P6]) # Left-ish
        C = dist.euclidean(face_landmarks[self.P7], face_landmarks[self.P8]) # Right-ish
        
        # Horizontal distance
        D = dist.euclidean(face_landmarks[self.P1], face_landmarks[self.P2])
        
        if D == 0:
            return 0.0

        mar = (A + B + C) / (3.0 * D)
        return mar
