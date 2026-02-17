
import mediapipe
print(f"Mediapipe file: {mediapipe.__file__}")
print(f"Dir(mediapipe): {dir(mediapipe)}")
try:
    import mediapipe.python.solutions
    print("Imported mediapipe.python.solutions successfully")
except ImportError as e:
    print(f"Failed to import mediapipe.python.solutions: {e}")
