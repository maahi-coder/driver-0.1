
import importlib.util
import sys

try:
    spec = importlib.util.find_spec("mediapipe.python.solutions")
    if spec:
        print(f"Found spec: {spec.origin}")
    else:
        print("Spec not found for mediapipe.python.solutions")
        
    import mediapipe
    print(f"Mediapipe path: {mediapipe.__path__}")
    
except Exception as e:
    print(e)
