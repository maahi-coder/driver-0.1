import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'eye_state_model.h5')
FACE_LANDMARKER_PATH = os.path.join(MODELS_DIR, 'face_landmarker.task')
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_FILE = os.path.join(DATA_DIR, 'logs.csv')
ALARM_FILE = os.path.join(BASE_DIR, 'utils', 'alarm.wav') # Placeholder, will need to be generated or provided

# Camera Settings
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Detection Thresholds
EYE_ASPECT_RATIO_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 20
YAWN_THRESH = 0.6  # MAR ratio threshold (0.6 is a good starting point)
YAWN_CONSEC_FRAMES = 20

# Fatigue Score Weights (Percentages)
FATIGUE_SCORE_WEIGHTS = {
    'eye': 0.40,
    'yawn': 0.30,
    'head': 0.20,
    'blink': 0.10
}

# Fatigue Alert Threshold
FATIGUE_SCORE_THRESHOLD = 70

# Head Pose Estimation
# 3D model points of face features (generic)
import numpy as np
FACE_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
