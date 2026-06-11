import ollama
import threading
import queue
# pyrefly: ignore [missing-import]
import cv2 as cv
import base64
import json 

class VLMWorker(threading.Thread):
    """Background AI worker. It runs on a separate thread to prevent camera freezing."""
    def __init__(self, host_ip="192.168.239.57", model_name="llama3.2-vision"):
        super().__init__(daemon=True)
        self.client = ollama.Client(host=f'http://{host_ip}:11434')
        self.model_name = model_name
        self.queue = queue.Queue()
        
        self.latest_insight = None  
        self.is_processing = False

    def run(self):
        while True:
            task = self.queue.get()
            if task is None: 
                break 
                
            frame, prompt, force_json, system_prompt = task
            self.is_processing = True
            
            try:
                _, buffer = cv.imencode('.jpg', frame)
                img_str = base64.b64encode(buffer).decode('utf-8')
                
                options = {"temperature": 0.0} if force_json else {}
                
                res = self.client.generate(
                    model=self.model_name, 
                    prompt=prompt, 
                    system=system_prompt,
                    images=[img_str],
                    format="json" if force_json else "",
                    options=options
                )
                
                raw_text = res.get('response', '')
                if force_json:
                    self.latest_insight = json.loads(raw_text) 
                    print(f"🤖 [VLM Routed Command]: {self.latest_insight}")
                else:
                    self.latest_insight = raw_text 
                    print(f"🧠 [VLM Thought]: {self.latest_insight}")
                    
            except Exception as e:
                print(f"[VLM ERROR]: {e}")
                self.latest_insight = {"error": str(e)}
            finally:
                self.is_processing = False


    def analyze_scene(self, frame, prompt="Describe the objects in the scene."):
        if not self.is_processing:
            self.queue.put((frame.copy(), prompt, False, ""))

    def route_command(self, frame, user_spoken_command, allowed_functions):
        if not self.is_processing:
            sys_prompt = f"""
            You are a robotic routing engine. Map the user's request to ONE of these functions: {allowed_functions}.
            Respond ONLY in JSON format. 
            Example: {{"function_name": "grab", "args": {{"target_color": "red"}}}}
            """
            self.queue.put((frame.copy(), user_spoken_command, True, sys_prompt))

    def stop(self):
        self.queue.put(None)
