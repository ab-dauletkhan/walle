"""
WALL-E Robot Control Tools
Complete set of tools for LLM to control the physical robot
Based on AI_Control_Documentation.md
"""

def get_robot_control_tools():
    """Returns all robot control tools for LLM function calling"""
    
    return [
        # ============= SERVO CONTROL TOOLS =============
        {
            "type": "function",
            "function": {
                "name": "set_head_rotation",
                "description": "Rotate WALL-E's head left or right. 0=far left, 50=center, 100=far right",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "position": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Head rotation position: 0=left, 50=center, 100=right"
                        }
                    },
                    "required": ["position"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_neck_position",
                "description": "Move WALL-E's neck up or down using both neck joints together",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_position": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Top neck joint position: 0=down, 100=up"
                        },
                        "bottom_position": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Bottom neck joint position: 0=down, 100=up"
                        }
                    },
                    "required": ["top_position", "bottom_position"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_both_eyes",
                "description": "Move both of WALL-E's eyes together to the same position",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "position": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Eye position: 0=looking down, 50=center, 100=looking up"
                        }
                    },
                    "required": ["position"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_individual_eyes",
                "description": "Move WALL-E's left and right eyes independently (for expressive looks)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "left_eye": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Left eye position: 0=down, 100=up"
                        },
                        "right_eye": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Right eye position: 0=down, 100=up"
                        }
                    },
                    "required": ["left_eye", "right_eye"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_both_arms",
                "description": "Move both of WALL-E's arms together to the same position",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "position": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Arm position: 0=down, 50=horizontal, 100=up"
                        }
                    },
                    "required": ["position"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_individual_arms",
                "description": "Move WALL-E's left and right arms independently",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "left_arm": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Left arm position: 0=down, 100=up"
                        },
                        "right_arm": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Right arm position: 0=down, 100=up"
                        }
                    },
                    "required": ["left_arm", "right_arm"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "look_at",
                "description": "Make WALL-E look in a specific direction by combining head rotation and neck position",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "horizontal": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Horizontal direction: 0=far left, 50=straight, 100=far right"
                        },
                        "vertical": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Vertical direction: 0=down, 50=level, 100=up"
                        }
                    },
                    "required": ["horizontal", "vertical"],
                    "additionalProperties": False
                }
            }
        },
        
        # ============= EMOTIONAL EXPRESSIONS =============
        {
            "type": "function",
            "function": {
                "name": "express_emotion",
                "description": "Make WALL-E express an emotion using preset servo positions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "emotion": {
                            "type": "string",
                            "enum": ["happy", "sad", "surprised", "neutral", "curious", "confused"],
                            "description": "The emotion to express"
                        }
                    },
                    "required": ["emotion"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "wave_hello",
                "description": "Make WALL-E wave hello with one arm (friendly greeting gesture)",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "reset_to_neutral",
                "description": "Reset all servos to neutral/center position (head center, neck level, arms down)",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            }
        },
        
        # ============= DRIVE MOTOR CONTROL =============
        {
            "type": "function",
            "function": {
                "name": "drive_forward",
                "description": "Drive WALL-E forward at specified speed",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "speed": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Speed percentage: 0=stop, 100=maximum speed"
                        },
                        "duration_ms": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Optional: How long to drive in milliseconds (0=continuous until stopped)"
                        }
                    },
                    "required": ["speed"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "drive_backward",
                "description": "Drive WALL-E backward at specified speed",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "speed": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Speed percentage: 0=stop, 100=maximum speed"
                        },
                        "duration_ms": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Optional: How long to drive in milliseconds (0=continuous until stopped)"
                        }
                    },
                    "required": ["speed"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "turn_left",
                "description": "Turn WALL-E left in place by rotating tracks in opposite directions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "speed": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Turn speed percentage: 0=no turn, 100=maximum turn rate"
                        },
                        "duration_ms": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Optional: How long to turn in milliseconds (0=continuous until stopped)"
                        }
                    },
                    "required": ["speed"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "turn_right",
                "description": "Turn WALL-E right in place by rotating tracks in opposite directions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "speed": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Turn speed percentage: 0=no turn, 100=maximum turn rate"
                        },
                        "duration_ms": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Optional: How long to turn in milliseconds (0=continuous until stopped)"
                        }
                    },
                    "required": ["speed"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "stop_movement",
                "description": "Stop all motor movement immediately (emergency stop)",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "move_with_differential",
                "description": "Advanced control: Set left and right track speeds independently for curved movement",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "left_speed": {
                            "type": "integer",
                            "minimum": -100,
                            "maximum": 100,
                            "description": "Left track speed: -100=full backward, 0=stop, 100=full forward"
                        },
                        "right_speed": {
                            "type": "integer",
                            "minimum": -100,
                            "maximum": 100,
                            "description": "Right track speed: -100=full backward, 0=stop, 100=full forward"
                        },
                        "duration_ms": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Optional: How long to move in milliseconds (0=continuous until stopped)"
                        }
                    },
                    "required": ["left_speed", "right_speed"],
                    "additionalProperties": False
                }
            }
        },
        
        # ============= COMPLEX BEHAVIORS =============
        {
            "type": "function",
            "function": {
                "name": "scan_surroundings",
                "description": "Make WALL-E scan surroundings by looking left, center, right (patrol behavior)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "speed": {
                            "type": "string",
                            "enum": ["slow", "normal", "fast"],
                            "description": "How fast to scan"
                        }
                    },
                    "required": ["speed"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "perform_greeting",
                "description": "Perform a full greeting sequence: look at person, wave, express happiness",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "navigate_to_position",
                "description": "Navigate to a relative position using combined movement (forward/backward + turns)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "distance_cm": {
                            "type": "integer",
                            "description": "Distance to move in centimeters (negative=backward)"
                        },
                        "angle_degrees": {
                            "type": "integer",
                            "minimum": -180,
                            "maximum": 180,
                            "description": "Angle to turn before moving (negative=left, positive=right)"
                        }
                    },
                    "required": ["distance_cm"],
                    "additionalProperties": False
                }
            }
        }
    ]


def get_robot_tool_names():
    """Returns list of all robot control tool names"""
    tools = get_robot_control_tools()
    return [tool["function"]["name"] for tool in tools]


class RobotControlExecutor:
    """
    Executes robot control commands by sending serial commands to Arduino
    This is a mock implementation - replace with actual serial communication
    """
    
    def __init__(self, serial_port=None):
        """
        Initialize robot controller
        
        Args:
            serial_port: Serial port object (e.g., serial.Serial('/dev/ttyUSB0', 115200))
                        If None, runs in simulation mode
        """
        self.serial_port = serial_port
        self.simulation_mode = serial_port is None
        # Registry mapping tool name -> handler callable
        # Each handler signature: handler(self, args: dict) -> str
        self._registry = {}
        self._register_default_handlers_deferred = False  # marker for deferred population

    # ---------------- Registry Utilities ----------------
    def register(self, name: str, handler):
        """Register a robot control handler.
        Args:
            name: Tool/function name exposed to LLM
            handler: Callable accepting (self, args) returning str result
        """
        if name in self._registry:
            # Allow override but note in simulation output if enabled
            if self.simulation_mode:
                print(f"[SIM][RobotRegistry] Overriding handler for '{name}'")
        self._registry[name] = handler

    def list_registered(self):
        """Return sorted list of registered robot tool names."""
        return sorted(self._registry.keys())

    def get_help(self):
        """Return a help string enumerating available robot tool handlers."""
        lines = ["Registered Robot Control Tools (" + str(len(self._registry)) + "):"]
        for name in self.list_registered():
            lines.append(f" - {name}")
        return "\n".join(lines)
        
    def send_command(self, command: str) -> str:
        """
        Send a command to the robot via serial
        
        Args:
            command: Command string to send (e.g., "G75" for head rotation)
            
        Returns:
            Status message
        """
        if self.simulation_mode:
            return f"[SIM] Sent command: {command}"
        
        try:
            self.serial_port.write(f"{command}\n".encode())
            return f"✓ Sent: {command}"
        except Exception as e:
            return f"❌ Serial error: {e}"
    
    def execute(self, fn_name: str, args: dict) -> str:
        """
        Execute a robot control function
        
        Args:
            fn_name: Name of the function to execute
            args: Arguments for the function
            
        Returns:
            Human-readable result message
        """
        
        # Populate registry on first use if empty
        if not self._registry and not self._register_default_handlers_deferred:
            self._populate_registry_from_legacy()
            self._register_default_handlers_deferred = True

        handler = self._registry.get(fn_name)
        if handler:
            try:
                return handler(self, args)
            except Exception as e:
                return f"❌ Handler error for {fn_name}: {e}"
        return f"❌ Unknown robot command: {fn_name}"

    # ---------------- Legacy -> Registry Migration ----------------
    def _populate_registry_from_legacy(self):
        """Populate registry with handler methods extracted from legacy if/elif chain.
        This provides backward compatibility until explicit handler methods are created.
        """
        # Register dedicated handler methods (new structure)
        self.register("set_head_rotation", lambda s, a: s._handle_set_head_rotation(a))
        self.register("set_neck_position", lambda s, a: s._handle_set_neck_position(a))
        self.register("set_both_eyes", lambda s, a: s._handle_set_both_eyes(a))
        self.register("set_individual_eyes", lambda s, a: s._handle_set_individual_eyes(a))
        self.register("set_both_arms", lambda s, a: s._handle_set_both_arms(a))
        self.register("set_individual_arms", lambda s, a: s._handle_set_individual_arms(a))
        self.register("look_at", lambda s, a: s._handle_look_at(a))
        self.register("express_emotion", lambda s, a: s._handle_express_emotion(a))
        self.register("wave_hello", lambda s, a: s._handle_wave_hello(a))
        self.register("reset_to_neutral", lambda s, a: s._handle_reset_to_neutral(a))
        self.register("drive_forward", lambda s, a: s._handle_drive_forward(a))
        self.register("drive_backward", lambda s, a: s._handle_drive_backward(a))
        self.register("turn_left", lambda s, a: s._handle_turn_left(a))
        self.register("turn_right", lambda s, a: s._handle_turn_right(a))
        self.register("stop_movement", lambda s, a: s._handle_stop_movement(a))
        self.register("move_with_differential", lambda s, a: s._handle_move_with_differential(a))
        self.register("scan_surroundings", lambda s, a: s._handle_scan_surroundings(a))
        self.register("perform_greeting", lambda s, a: s._handle_perform_greeting(a))
        self.register("navigate_to_position", lambda s, a: s._handle_navigate_to_position(a))

    # ---------------- Legacy Handler Implementations ----------------
    # Each of these is a direct extraction of code from the original if/elif branches.
    # ---------------- New Handler Methods ----------------
    def _handle_set_head_rotation(self, args):
        position = args.get("position", 50)
        cmd = f"G{position}"
        self.send_command(cmd)
        direction = "left" if position < 40 else "right" if position > 60 else "center"
        return f"🤖 Head rotated to {position}% ({direction})"

    def _handle_set_neck_position(self, args):
        top = args.get("top_position", 50)
        bottom = args.get("bottom_position", 50)
        self.send_command(f"N{top}")
        self.send_command(f"M{bottom}")
        return f"🤖 Neck positioned: top={top}%, bottom={bottom}%"

    def _handle_set_both_eyes(self, args):
        position = args.get("position", 50)
        self.send_command(f"L{position}")
        self.send_command(f"R{position}")
        looking = "up" if position > 60 else "down" if position < 40 else "straight"
        return f"👀 Eyes looking {looking} (position={position}%)"

    def _handle_set_individual_eyes(self, args):
        left = args.get("left_eye", 50)
        right = args.get("right_eye", 50)
        self.send_command(f"L{left}")
        self.send_command(f"R{right}")
        return f"👀 Eyes positioned: left={left}%, right={right}%"

    def _handle_set_both_arms(self, args):
        position = args.get("position", 50)
        self.send_command(f"A{position}")
        self.send_command(f"B{position}")
        arm_pos = "raised" if position > 60 else "lowered" if position < 40 else "neutral"
        return f"💪 Arms {arm_pos} (position={position}%)"

    def _handle_set_individual_arms(self, args):
        left = args.get("left_arm", 50)
        right = args.get("right_arm", 50)
        self.send_command(f"A{left}")
        self.send_command(f"B{right}")
        return f"💪 Arms positioned: left={left}%, right={right}%"

    def _handle_look_at(self, args):
        horizontal = args.get("horizontal", 50)
        vertical = args.get("vertical", 50)
        self.send_command(f"G{horizontal}")
        self.send_command(f"N{vertical}")
        h_dir = "left" if horizontal < 40 else "right" if horizontal > 60 else "straight"
        v_dir = "up" if vertical > 60 else "down" if vertical < 40 else "level"
        return f"🤖 Looking {h_dir} and {v_dir}"

    def _handle_express_emotion(self, args):
        emotion = args.get("emotion", "neutral")
        emotion_map = {
            "happy": [(80, 80), (60, 60), (70, 70)],
            "sad": [(20, 20), (40, 40), (20, 20)],
            "surprised": [(100, 100), (80, 80), (50, 50)],
            "neutral": [(50, 50), (50, 50), (40, 40)],
            "curious": [(60, 70), (60, 60), (40, 40)],
            "confused": [(40, 60), (55, 55), (45, 45)]
        }
        if emotion in emotion_map:
            eyes, neck, arms = emotion_map[emotion]
            self.send_command(f"L{eyes[0]}")
            self.send_command(f"R{eyes[1]}")
            self.send_command(f"N{neck[0]}")
            self.send_command(f"A{arms[0]}")
            self.send_command(f"B{arms[1]}")
            return f"😊 Expressing emotion: {emotion}"
        return f"❌ Unknown emotion: {emotion}"

    def _handle_wave_hello(self, args):
        for pos in [70, 80, 70, 80, 70, 40]:
            self.send_command(f"B{pos}")
        return f"👋 Waved hello!"

    def _handle_reset_to_neutral(self, args):
        self.send_command("G50")
        self.send_command("N50")
        self.send_command("L50")
        self.send_command("R50")
        self.send_command("A40")
        self.send_command("B40")
        return f"↺ Reset to neutral position"

    def _handle_drive_forward(self, args):
        speed = args.get("speed", 50)
        duration = args.get("duration_ms", 0)
        self.send_command(f"Y{speed}")
        msg = f"🤖 Driving forward at {speed}% speed"
        if duration > 0:
            msg += f" for {duration}ms"
        return msg

    def _handle_drive_backward(self, args):
        speed = args.get("speed", 50)
        duration = args.get("duration_ms", 0)
        self.send_command(f"Y{-speed}")
        msg = f"🤖 Driving backward at {speed}% speed"
        if duration > 0:
            msg += f" for {duration}ms"
        return msg

    def _handle_turn_left(self, args):
        speed = args.get("speed", 50)
        duration = args.get("duration_ms", 0)
        self.send_command(f"X{-speed}")
        msg = f"🤖 Turning left at {speed}% speed"
        if duration > 0:
            msg += f" for {duration}ms"
        return msg

    def _handle_turn_right(self, args):
        speed = args.get("speed", 50)
        duration = args.get("duration_ms", 0)
        self.send_command(f"X{speed}")
        msg = f"🤖 Turning right at {speed}% speed"
        if duration > 0:
            msg += f" for {duration}ms"
        return msg

    def _handle_stop_movement(self, args):
        self.send_command("q")
        return f"🛑 All movement stopped"

    def _handle_move_with_differential(self, args):
        left = args.get("left_speed", 0)
        right = args.get("right_speed", 0)
        duration = args.get("duration_ms", 0)
        forward = (left + right) // 2
        turn = (right - left) // 2
        self.send_command(f"Y{forward}")
        self.send_command(f"X{turn}")
        msg = f"🤖 Differential drive: L={left}%, R={right}%"
        if duration > 0:
            msg += f" for {duration}ms"
        return msg

    def _handle_scan_surroundings(self, args):
        speed = args.get("speed", "normal")
        delay_map = {"slow": 1500, "normal": 800, "fast": 400}
        delay = delay_map.get(speed, 800)
        for pos in [20, 50, 80, 50]:
            self.send_command(f"G{pos}")
        return f"👀 Scanned surroundings at {speed} speed"

    def _handle_perform_greeting(self, args):
        self.send_command("G60")
        self.send_command("N60")
        self.send_command("L80")
        self.send_command("R80")
        for pos in [70, 80, 70, 80, 70, 40]:
            self.send_command(f"B{pos}")
        return f"👋 Performed greeting sequence"

    def _handle_navigate_to_position(self, args):
        distance = args.get("distance_cm", 0)
        angle = args.get("angle_degrees", 0)
        messages = []
        if angle != 0:
            turn_cmd = f"X{abs(angle)//2}" if angle > 0 else f"X{-abs(angle)//2}"
            self.send_command(turn_cmd)
            messages.append(f"Turned {angle}°")
        if distance != 0:
            speed = min(abs(distance), 100)
            move_cmd = f"Y{speed}" if distance > 0 else f"Y{-speed}"
            self.send_command(move_cmd)
            messages.append(f"Moved {distance}cm")
        return f"🤖 Navigation: {', '.join(messages)}"

# External utility wrappers for introspection (similar to memory tools style)
def list_registered_robot_tools(executor: RobotControlExecutor) -> list:
    """Return list of registered robot tool names from given executor."""
    return executor.list_registered()

def get_robot_tools_help(executor: RobotControlExecutor) -> str:
    """Return help text enumerating robot tools."""
    return executor.get_help()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("🤖 WALL-E Robot Control Tools")
    print("=" * 70)
    
    tools = get_robot_control_tools()
    print(f"\n✓ Loaded {len(tools)} robot control tools:")
    
    categories = {
        "Servo Control": ["set_head_rotation", "set_neck_position", "set_both_eyes", 
                          "set_individual_eyes", "set_both_arms", "set_individual_arms", "look_at"],
        "Emotions": ["express_emotion", "wave_hello", "reset_to_neutral"],
        "Movement": ["drive_forward", "drive_backward", "turn_left", "turn_right", 
                    "stop_movement", "move_with_differential"],
        "Behaviors": ["scan_surroundings", "perform_greeting", "navigate_to_position"]
    }
    
    for category, tool_names in categories.items():
        print(f"\n{category}:")
        for tool_name in tool_names:
            print(f"  • {tool_name}")
    
    print("\n" + "=" * 70)
    print("\n💡 Testing executor in simulation mode...")
    
    executor = RobotControlExecutor()  # Simulation mode
    # Force registry population
    executor.execute("set_head_rotation", {"position": 50})  # triggers populate if empty
    print("\n" + executor.get_help())
    
    test_commands = [
        ("set_head_rotation", {"position": 75}),
        ("express_emotion", {"emotion": "happy"}),
        ("drive_forward", {"speed": 50}),
        ("wave_hello", {}),
    ]
    
    for fn_name, args in test_commands:
        result = executor.execute(fn_name, args)
        print(f"  {result}")
    
    print("\n✓ Robot tools ready for integration!")
