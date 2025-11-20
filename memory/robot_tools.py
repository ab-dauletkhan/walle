"""
WALL-E Robot Control Tools
Updated with blocking execution for realistic timing.
"""
import time

def get_robot_control_tools():
    # (Schema definitions same as before, omitted for brevity but assume full JSON list here)
    # Refer to previous upload for exact JSON schema
    return [
        {"type": "function", "function": {"name": "set_head_rotation", "parameters": {"type": "object", "properties": {"position": {"type": "integer"}}, "required": ["position"]}}},
        {"type": "function", "function": {"name": "drive_forward", "parameters": {"type": "object", "properties": {"speed": {"type": "integer"}, "duration_ms": {"type": "integer"}}, "required": ["speed"]}}},
        {"type": "function", "function": {"name": "express_emotion", "parameters": {"type": "object", "properties": {"emotion": {"type": "string"}}, "required": ["emotion"]}}},
        {"type": "function", "function": {"name": "scan_surroundings", "parameters": {"type": "object", "properties": {"speed": {"type": "string"}}, "required": ["speed"]}}},
        # ... add remaining tools from original ...
    ]

def get_robot_tool_names():
    return ["set_head_rotation", "drive_forward", "drive_backward", "turn_left", 
            "turn_right", "stop_movement", "express_emotion", "scan_surroundings",
            "wave_hello", "set_both_arms", "set_neck_position", "reset_to_neutral"]

class RobotControlExecutor:
    def __init__(self, serial_port=None):
        self.serial_port = serial_port
        self.simulation = serial_port is None

    def send_command(self, cmd: str):
        if self.simulation:
            print(f"🔌 [SERIAL SIM] >> {cmd}")
        else:
            try:
                self.serial_port.write(f"{cmd}\n".encode())
            except Exception as e:
                return f"Serial Error: {e}"

    def execute(self, fn_name: str, args: dict) -> str:
        method = getattr(self, f"_handle_{fn_name}", None)
        if method:
            return method(args)
        return f"❌ Unknown command: {fn_name}"

    # --- Handlers with Blocking Time ---
    
    def _handle_drive_forward(self, args):
        speed = args.get("speed", 50)
        duration = args.get("duration_ms", 0)
        self.send_command(f"Y{speed}")
        
        if duration > 0:
            # Block execution so LLM knows time is passing
            time.sleep(duration / 1000.0)
            self.send_command("q") # Auto stop after duration
            return f"🤖 Drove forward at {speed}% for {duration}ms"
        return f"🤖 Driving forward at {speed}% (Continuous)"

    def _handle_turn_left(self, args):
        speed = args.get("speed", 50)
        duration = args.get("duration_ms", 0)
        self.send_command(f"X{-speed}")
        if duration > 0:
            time.sleep(duration / 1000.0)
            self.send_command("q")
            return f"🤖 Turned left at {speed}% for {duration}ms"
        return f"🤖 Turning left at {speed}%"

    def _handle_scan_surroundings(self, args):
        speed = args.get("speed", "normal")
        delay = {"slow": 1.5, "normal": 0.8, "fast": 0.4}.get(speed, 0.8)
        
        self.send_command("G20"); time.sleep(delay)
        self.send_command("G50"); time.sleep(delay)
        self.send_command("G80"); time.sleep(delay)
        self.send_command("G50"); 
        return f"👀 Scanned surroundings ({speed})"

    def _handle_express_emotion(self, args):
        emotion = args.get("emotion", "neutral")
        # (Simplified mapping)
        map_e = {"happy": ("L80\nR80", "N80"), "sad": ("L20\nR20", "N20")}
        cmds = map_e.get(emotion, ("L50\nR50", "N50"))
        self.send_command(cmds[0])
        self.send_command(cmds[1])
        return f"😊 Expression set to {emotion}"

    def _handle_set_head_rotation(self, args):
        pos = args.get("position", 50)
        self.send_command(f"G{pos}")
        return f"Head to {pos}"
    
    def _handle_wave_hello(self, args):
        for _ in range(2):
            self.send_command("B80"); time.sleep(0.3)
            self.send_command("B40"); time.sleep(0.3)
        return "👋 Waved hello"
        
    def _handle_reset_to_neutral(self, args):
        self.send_command("G50"); self.send_command("N50"); 
        self.send_command("L50"); self.send_command("R50");
        return "Neutral position"
    
    def _handle_stop_movement(self, args):
        self.send_command("q")
        return "🛑 Emergency Stop"