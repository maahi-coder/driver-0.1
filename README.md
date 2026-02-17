# AI-Based Adaptive Driver Cognitive Monitoring System

A production-ready system to monitor driver fatigue and cognitive state using real-time computer vision and AI.

## Features

*   **Real-time Monitoring**: Uses webcam feed to detect driver state.
*   **AI-Powered Eye State Detection**: Uses MobileNetV2 (Transfer Learning) to classify open/closed eyes.
*   **Behavioral Metrics**:
    *   Eye Aspect Ratio (EAR)
    *   Mouth Aspect Ratio (MAR) for yawn detection
    *   Head Pose Estimation (Pitch/Yaw/Roll)
*   **Fatigue Scoring**: Weighted algorithm combining multiple factors to assess drowsiness.
*   **Audio Alerts**: Triggers alarm when fatigue threshold is breached.
*   **Live Dashboard**: Streamlit-based dashboard for visualizing historical data and trends.

## Project Structure

```
driver_monitoring/
├── run.py                 # Main application point
├── start.sh               # Startup script (sets up env)
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
├── config/
│   └── settings.py        # Configuration constants
├── models/
│   └── train_model.py     # Script to train AI model
├── utils/                 # Helper modules
│   ├── face_detection.py  
│   ├── eye_detection.py
│   ├── yawn_detection.py
│   ├── head_pose.py
│   ├── fatigue_score.py
│   ├── alarm.py
│   └── logger.py
├── dashboard/
│   └── app.py             # Streamlit dashboard
└── data/
    └── logs.csv           # Event logs
```

## Installation & Usage

### Prerequisites
*   Python 3.8+
*   Webcam

### Setup

1.  **Clone/Navigate to directory**:
    ```bash
    cd driver_monitoring
    ```

2.  **Run the Startup Script**:
    This script will automatically create a virtual environment, install dependencies, and run the application.
    ```bash
    ./start.sh
    ```
    
    **Troubleshooting:**
    If you encounter permission issues, run:
    ```bash
    chmod +x start.sh
    ./start.sh
    ```

    *Alternatively, manual setup:*
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python run.py
    ```

### Training the Model (Optional)

The system works with a default EAR logic if no model is present. To train the AI model for better accuracy:

1.  Place your dataset in `data/dataset_B_Eye_Images` with `open` and `closed` subfolders.
2.  Run the training script:
    ```bash
    python models/train_model.py
    ```

### Running the Dashboard

To view the live analytics dashboard, run:

```bash
streamlit run dashboard/app.py
```

## Configuration

You can adjust thresholds and weights in `config/settings.py`:
*   `FATIGUE_SCORE_THRESHOLD`: Trigger level for alarm (default: 70)
*   `FATIGUE_SCORE_WEIGHTS`: Importance of different factors (Eye, Yawn, Head, Blink)

## Future Improvements

*   Implement blink rate calculation with history tracking.
*   Integrate more advanced Gaze Tracking.
*   Deploy to edge devices (Raspberry Pi/Jetson Nano).
*   Cloud integration for fleet management.
