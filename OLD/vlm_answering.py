import json
import requests
import base64

# ==========================================
# 1. Define Scene Query Functions
# ==========================================
def analyze_stacking():
    print("🔍 Executing: analyze_stacking")
    # In a real app, this function would ask the VLM specifically about stacks
    return "Action triggered: Analyzing vertical stacking order."

def check_clearance(target_color):
    print(f"🔍 Executing: check_clearance for '{target_color}'")
    # Ask the VLM if the top of the target_color block is clear
    return f"Action triggered: Checking if {target_color} block is obstructed."

def locate_objects_on_markers():
    print("🔍 Executing: locate_objects_on_markers")
    # Ask the VLM which blocks are placed on ArUco markers
    return "Action triggered: Finding blocks on markers."

# ==========================================
# 2. Create the Function Registry
# ==========================================
function_registry = {
    "analyze_stacking": analyze_stacking,
    "check_clearance": check_clearance,
    "locate_objects_on_markers": locate_objects_on_markers
}

# ==========================================
# 3. Helper: Encode Image for Ollama
# ==========================================
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ==========================================
# 4. The VLM Routing Engine
# ==========================================
def query_scene_with_vlm(user_instruction, image_path, model="llama3.2-vision"):
    """
    Sends the user text AND the image to the Vision model.
    Routes the intent to the correct function.
    """
    available_functions = list(function_registry.keys())
    base64_image = encode_image(image_path)
    
    system_prompt = f"""
    You are a robotic vision routing engine. Look at the provided image of the robot workspace.
    Analyze the user's request and map it to ONE of these functions:
    {available_functions}
    
    If the user asks about blocks resting on other blocks, use 'analyze_stacking'.
    If the user asks if a specific block can be picked up safely, use 'check_clearance' and extract the color.
    If the user asks about blocks on the black and white square markers, use 'locate_objects_on_markers'.
    
    You must respond ONLY with a JSON object containing "function_name" and an optional "args" dictionary.
    
    Example: {{"function_name": "check_clearance", "args": {{"target_color": "pink"}}}}
    """

    payload = {
        "model": model,
        "prompt": f"User request: {user_instruction}",
        "system": system_prompt,
        "images": [base64_image], # Passing the image to the VLM
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0
        }
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        
        llm_output = json.loads(response.json().get("response", "{}"))
        
        func_name = llm_output.get("function_name")
        args = llm_output.get("args", {})
        
        print(f"🤖 VLM routed request to: '{func_name}' with args: {args}")

        # Execute the matched function
        if func_name in function_registry:
            # Pass arguments if the function expects them
            if args and "target_color" in args and func_name == "check_clearance":
                function_registry[func_name](args["target_color"])
            else:


query_scene_with_vlm(user_instruction="Can I pick up the pink block safely?", image_path="workspace.jpg")