import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        # Approximate camera matrix if not provided
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion

    def get_pose(self, face_landmarks, image_shape, model_points):
        """
        Estimates head pose (pitch, yaw, roll) from face landmarks.
        Args:
            face_landmarks: List of (x, y) tuples.
            image_shape: (height, width) of the image.
            model_points: 3D model points of the face.
        Returns:
            rotation_vector, translation_vector, angles (pitch, yaw, roll)
        """
        img_h, img_w = image_shape[:2]
        
        if self.camera_matrix is None:
            focal_length = img_w
            center = (img_w / 2, img_h / 2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

        # 2D Image Points
        # Must match the order of FACE_3D_MODEL_POINTS in settings.py
        # Nose, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth
        image_points = np.array([
            face_landmarks[1],    # Nose tip
            face_landmarks[152],  # Chin
            face_landmarks[33],   # Left eye left corner (approx) - actually let's use check settings.py
            face_landmarks[263],  # Right eye right corner
            face_landmarks[61],   # Left Mouth corner
            face_landmarks[291]   # Right mouth corner
        ], dtype="double")
        
        # Verify indices against settings.py
        # settings.py: 
        # (0.0, 0.0, 0.0),             # Nose tip
        # (0.0, -330.0, -65.0),        # Chin
        # (-225.0, 170.0, -135.0),     # Left eye left corner
        # (225.0, 170.0, -135.0),      # Right eye right corner
        # (-150.0, -150.0, -125.0),    # Left Mouth corner
        # (150.0, -150.0, -125.0)      # Right mouth corner

        # Corresponding MediaPipe indices (approximate):
        # Nose tip: 1
        # Chin: 152
        # Left Eye Left Corner: 33
        # Right Eye Right Corner: 263
        # Left Mouth Corner: 61
        # Right Mouth Corner: 291

        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, 
            image_points, 
            self.camera_matrix, 
            self.dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        return success, rotation_vector, translation_vector

    def get_angles(self, rotation_vector, translation_vector):
        """
        Converts rotation vector to Euler angles (pitch, yaw, roll).
        """
        rmat, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate Euler angles
        # pitch: x-axis rotation, yaw: y-axis rotation, roll: z-axis rotation
        # A bit complex, simplified:
        proj_matrix = np.hstack((rmat, translation_vector))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        
        pitch, yaw, roll = [np.squeeze(x) for x in euler_angles]
        
        return pitch, yaw, roll
