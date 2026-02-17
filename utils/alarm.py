import pygame
import os
import threading
import time

class AlarmSystem:
    def __init__(self, sound_file):
        """
        Args:
            sound_file: Path to the .wav or .mp3 file.
        """
        self.sound_file = sound_file
        self.playing = False
        try:
            pygame.mixer.init()
            if os.path.exists(sound_file):
                self.sound = pygame.mixer.Sound(sound_file)
            else:
                print(f"Warning: Sound file {sound_file} not found. Audio alerts disabled.")
                self.sound = None
        except Exception as e:
            print(f"Error initializing Pygame mixer: {e}")
            self.sound = None

    def play(self):
        """
        Plays the alarm sound in a separate thread to avoid blocking.
        """
        if self.sound and not self.playing:
            self.playing = True
            threading.Thread(target=self._play_thread).start()

    def _play_thread(self):
        try:
            self.sound.play()
            # Wait for sound to finish or just let it play once
            while pygame.mixer.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Error playing sound: {e}")
        finally:
            self.playing = False

    def stop(self):
        if self.sound:
            self.sound.stop()
            self.playing = False
