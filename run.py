import cv2
import numpy as np
import time
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import settings
    from utils.face_detection import FaceDetector
    from utils.eye_detection import EyeDetector
    from utils.yawn_detection import YawnDetector
    from utils.head_pose import HeadPoseEstimator
    from utils.fatigue_score import FatigueScorer
    from utils.alarm import AlarmSystem
    from utils.logger import get_logger
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Initialize Logger
logger = get_logger("DriverMonitor")

class DriverMonitorSystem:
    def __init__(self):
        self.running = False
        self.cap = None
        
        # Initialize Modules
        self.face_detector = FaceDetector()
        self.eye_detector = EyeDetector()
        self.yawn_detector = YawnDetector()
        self.head_pose_estimator = HeadPoseEstimator()
        
        # Initialize Scorer
        self.fatigue_scorer = FatigueScorer(settings.FATIGUE_SCORE_WEIGHTS)
        
        # Initialize Alarm
        self.alarm = AlarmSystem(settings.ALARM_FILE)
        
        # Load Model
        self.model = self._load_model()
        
        # State Variables
        self.fatigue_score = 0
        self.alert_active = False
        self.eye_closed_frames = 0
        self.yawn_frames = 0
        self.total_frames = 0
        
        # File Logging
        self.log_file = open(settings.LOG_FILE, 'a')
        if os.stat(settings.LOG_FILE).st_size == 0:
            self.log_file.write("timestamp,fatigue_score,alert_active,eye_state,yawn_detected,head_pitch,head_yaw\n")

    def _load_model(self):
        """Loads the trained eye state model or creates a dummy one."""
        if os.path.exists(settings.MODEL_PATH):
            try:
                logger.info(f"Loading model from {settings.MODEL_PATH}")
                return load_model(settings.MODEL_PATH)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return None
        else:
            logger.warning("Model file not found. Using fallback detection (EAR only).")
            # Suggest training
            logger.info("Please run 'python models/train_model.py' first to generate a model.")
            return None

    def predict_eye_state(self, eye_img):
        """
        Predicts if eye is open or closed using the model.
        Returns: 0 for closed, 1 for open (or confidence).
        """
        if self.model is None:
            # Fallback to EAR (handled in main loop)
            return None
            
        try:
            # Preprocess
            img = tf.cast(eye_img, tf.float32)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            
            prediction = self.model.predict(img, verbose=0)
            return prediction[0][0]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

    def run(self):
        self.cap = cv2.VideoCapture(settings.CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            logger.error("Could not open camera.")
            return

        self.running = True
        logger.info("Starting monitoring system...")
        
        prev_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(frame)
            
            # Default values for metrics
            ear = 0
            mar = 0
            pitch, yaw, roll = 0, 0, 0
            eye_closed = False
            yawn_detected = False
            
            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    # Draw Mesh
                    self.face_detector.draw_landmarks(frame, results)
                    
                    # Get landmarks as numpy array
                    lm_np = self.face_detector.get_landmarks_as_np(results, frame.shape[1], frame.shape[0])
                    
                    if lm_np is None:
                        continue

                    # 1. Eye Detection
                    left_eye, right_eye = self.eye_detector.get_eye_landmarks(lm_np)
                    left_ear = self.eye_detector.calculate_ear(left_eye)
                    right_ear = self.eye_detector.calculate_ear(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    
                    # Model Inference (Optional / Hybrid)
                    # Use model if available, otherwise use EAR threshold
                    if self.model:
                        # Extract eye regions
                        left_eye_img = self.eye_detector.extract_eye_region(frame, left_eye)
                        right_eye_img = self.eye_detector.extract_eye_region(frame, right_eye)
                        
                        p_left = self.predict_eye_state(left_eye_img)
                        p_right = self.predict_eye_state(right_eye_img)
                        
                        if p_left is not None and p_right is not None:
                            # Assuming model outputs probability of OPEN
                            # If prob < 0.5, it's CLOSED
                            if p_left < 0.5 and p_right < 0.5:
                                eye_closed = True
                        else:
                            # Fallback if prediction failed
                            if ear < settings.EYE_ASPECT_RATIO_THRESHOLD:
                                eye_closed = True
                    else:
                        # Fallback to EAR
                        if ear < settings.EYE_ASPECT_RATIO_THRESHOLD:
                            eye_closed = True

                    if eye_closed:
                        self.eye_closed_frames += 1
                    else:
                        self.eye_closed_frames = 0
                        
                    # 2. Yawn Detection
                    mar = self.yawn_detector.calculate_mar(lm_np)
                    
                    if mar > settings.YAWN_THRESH: # Use centralized setting
                         self.yawn_frames += 1
                         yawn_detected = True
                    else:
                         self.yawn_frames = 0

                    # 3. Head Pose
                    success, rot_vec, trans_vec = self.head_pose_estimator.get_pose(lm_np, frame.shape, settings.FACE_3D_MODEL_POINTS)
                    if success:
                        pitch, yaw, roll = self.head_pose_estimator.get_angles(rot_vec, trans_vec)

            # --- Fatigue Calculation ---
            # Create inputs for scorer (normalize to 0-1)
            # Eye closure duration: e.g. > 1 sec (30 frames) is bad
            norm_eye = min(1.0, self.eye_closed_frames / settings.EYE_AR_CONSEC_FRAMES)
            
            # Yawn frequency: Not tracking history deeply here, just instantaneous state
            norm_yawn = 1.0 if self.yawn_frames > settings.YAWN_CONSEC_FRAMES else 0.0
            
            # Head tilt: Normalize pitch/yaw. e.g. > 20 degrees is bad
            norm_head = min(1.0, (abs(pitch) + abs(yaw)) / 40.0)
            
            # Blink rate: (Simplified: we aren't tracking blink RATE, just closure)
            norm_blink = 0.0 # Placeholder
            
            self.fatigue_score = self.fatigue_scorer.calculate_score(norm_eye, norm_yawn, norm_head, norm_blink)
            
            # --- Alert ---
            if self.fatigue_score > settings.FATIGUE_SCORE_THRESHOLD:
                if not self.alert_active:
                    self.alert_active = True
                    self.alarm.play()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                self.alert_active = False
                self.alarm.stop()

            # --- GUI Overlay ---
            self._draw_overlay(frame, ear, mar, pitch, yaw)

            # --- Logging ---
            if time.time() - prev_time > 1.0: # Log every second
                self.log_file.write(f"{time.time()},{self.fatigue_score},{int(self.alert_active)},{'Closed' if eye_closed else 'Open'},{int(yawn_detected)},{pitch:.2f},{yaw:.2f}\n")
                self.log_file.flush()
                prev_time = time.time()

            cv2.imshow("Driver Monitoring System", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False

        self.cleanup()

    def _draw_overlay(self, frame, ear, mar, pitch, yaw):
        # Background for text
        cv2.rectangle(frame, (0, 0), (250, 160), (0, 0, 0), -1)
        
        cv2.putText(frame, f"Fatigue Score: {self.fatigue_score:.1f}%", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.fatigue_score < 50 else (0, 0, 255), 2)
        
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Head: P{pitch:.0f}/Y{yaw:.0f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if self.model is None:
             cv2.putText(frame, "Mode: Fallback (EAR)", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
             cv2.putText(frame, "Mode: AI Model", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    def cleanup(self):
        logger.info("Cleaning up...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.log_file.close()

if __name__ == "__main__":
    app = DriverMonitorSystem()
    app.run()
