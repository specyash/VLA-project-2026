import threading
import queue
import time
import os
import wave
import platform
import subprocess

try:
    from piper import PiperVoice
    piper_ready= True
except ImportError:
    piper_ready = False

class VoiceWorker(threading.Thread):
    def __init__(self, model_name="en_GB-alba-medium.onnx"):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self.active = False
        self.voice = None

        if not piper_ready:
            print("[VOICE INIT ERROR]: 'piper-tts' is not installed. Please run: pip install piper-tts")
            return

        # Define potential paths where the model file might exist
        potential_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "voice", model_name), # data/voice/
            os.path.join(os.path.dirname(__file__), model_name), # same folder as this script (src/voice/)
            model_name # current working directory
        ]

        # Find the first path that actually contains the model file
        self.model_path = None
        for path in potential_paths:
            if os.path.exists(path):
                self.model_path = path
                break

        if not self.model_path:
            print("\n[VOICE INIT ERROR]: Could not find the voice model file.")
            print(f"Please place '{model_name}' (and its corresponding '.json' file) in one of these directories: ")
            for path in potential_paths:
                print(f" -{os.path.abspath(path)}")
            return

        try:
            print(f"[Voice] Loading Piper model from: {self.model_path}")
            self.voice = PiperVoice.load(self.model_path)
            self.active = True
            print("[Voice] Piper TTS engine initialized successfully.")
        except Exception as e:
            print(f"[VOICE INIT ERROR]: Failed to load Piper model:{e}")

    def speak(self, text):

        if self.active:
            self.queue.put(text)
        print(f"[AI]: {text}")

    def play_audio(self, filename):
        try:
            if platform.system() == "Darwin": #for macos systems
                subprocess.run(["afplay", filename])
            elif platform.system() == "Windows": # for windows system
                import winsound
                winsound.PlaySound(filename, winsound.SND_FILENAME)
            else: # for linux system
                subprocess.run(["aplay", filename])
        except Exception as e:
            print(f"[VOICE PLAYBACK ERROR]: {e}")

    def run(self):
        if not self.active or self.voice is None:
            print("[Voice] Background thread did not start: voice engine is inactive.")
            return

        while True:
            text = self.queue.get()
            
            # None will be used as the termination signal
            if text is None:
                break
                
            try:
                # Generate a unique temporary WAV file name to avoid collisions
                filename = f"temp_piper_{int(time.time() * 1000)}.wav"
                
                # rendering speech->wav file
                with wave.open(filename, "wb") as wav_file:
                    self.voice.synthesize_wav(text, wav_file)
                
                # audio will now play
                self.play_audio(filename)
                
                # now we cleanup the temp files
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                print(f"[VOICE WORKER ERROR]: {e}")
            # giving out small timeouts to avoid cpu load
            time.sleep(0.1) 
            self.queue.task_done()

