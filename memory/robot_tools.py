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
        
        # ============= SERVO CONTROL =============
        if fn_name == "set_head_rotation":
            position = args.get("position", 50)
            cmd = f"G{position}"
            self.send_command(cmd)
            direction = "left" if position < 40 else "right" if position > 60 else "center"
            return f"🤖 Head rotated to {position}% ({direction})"
        
        elif fn_name == "set_neck_position":
            top = args.get("top_position", 50)
            bottom = args.get("bottom_position", 50)
            self.send_command(f"N{top}")
            self.send_command(f"M{bottom}")
            return f"🤖 Neck positioned: top={top}%, bottom={bottom}%"
        
        elif fn_name == "set_both_eyes":
            position = args.get("position", 50)
            self.send_command(f"L{position}")
            self.send_command(f"R{position}")
            looking = "up" if position > 60 else "down" if position < 40 else "straight"
            return f"👀 Eyes looking {looking} (position={position}%)"
        
        elif fn_name == "set_individual_eyes":
            left = args.get("left_eye", 50)
            right = args.get("right_eye", 50)
            self.send_command(f"L{left}")
            self.send_command(f"R{right}")
            return f"👀 Eyes positioned: left={left}%, right={right}%"
        
        elif fn_name == "set_both_arms":
            position = args.get("position", 50)
            self.send_command(f"A{position}")
            self.send_command(f"B{position}")
            arm_pos = "raised" if position > 60 else "lowered" if position < 40 else "neutral"
            return f"💪 Arms {arm_pos} (position={position}%)"
        
        elif fn_name == "set_individual_arms":
            left = args.get("left_arm", 50)
            right = args.get("right_arm", 50)
            self.send_command(f"A{left}")
            self.send_command(f"B{right}")
            return f"💪 Arms positioned: left={left}%, right={right}%"
        
        elif fn_name == "look_at":
            horizontal = args.get("horizontal", 50)
            vertical = args.get("vertical", 50)
            self.send_command(f"G{horizontal}")
            self.send_command(f"N{vertical}")
            h_dir = "left" if horizontal < 40 else "right" if horizontal > 60 else "straight"
            v_dir = "up" if vertical > 60 else "down" if vertical < 40 else "level"
            return f"🤖 Looking {h_dir} and {v_dir}"
        
        # ============= EMOTIONS =============
        elif fn_name == "express_emotion":
            emotion = args.get("emotion", "neutral")
            emotion_map = {
                "happy": [(80, 80), (60, 60), (70, 70)],  # eyes up, neck up, arms mid
                "sad": [(20, 20), (40, 40), (20, 20)],     # eyes down, neck down, arms down
                "surprised": [(100, 100), (80, 80), (50, 50)],  # eyes wide up, neck up
                "neutral": [(50, 50), (50, 50), (40, 40)],
                "curious": [(60, 70), (60, 60), (40, 40)],  # slightly asymmetric eyes
                "confused": [(40, 60), (55, 55), (45, 45)]  # very asymmetric eyes
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
        
        elif fn_name == "wave_hello":
            # Wave animation: raise right arm, move side to side, lower
            for pos in [70, 80, 70, 80, 70, 40]:
                self.send_command(f"B{pos}")
            return f"👋 Waved hello!"
        
        elif fn_name == "reset_to_neutral":
            self.send_command("G50")  # Head center
            self.send_command("N50")  # Neck level
            self.send_command("L50")  # Eyes center
            self.send_command("R50")
            self.send_command("A40")  # Arms down
            self.send_command("B40")
            return f"↺ Reset to neutral position"
        
        # ============= MOVEMENT =============
        elif fn_name == "drive_forward":
            speed = args.get("speed", 50)
            duration = args.get("duration_ms", 0)
            self.send_command(f"Y{speed}")
            msg = f"🤖 Driving forward at {speed}% speed"
            if duration > 0:
                msg += f" for {duration}ms"
            return msg
        
        elif fn_name == "drive_backward":
            speed = args.get("speed", 50)
            duration = args.get("duration_ms", 0)
            self.send_command(f"Y{-speed}")
            msg = f"🤖 Driving backward at {speed}% speed"
            if duration > 0:
                msg += f" for {duration}ms"
            return msg
        
        elif fn_name == "turn_left":
            speed = args.get("speed", 50)
            duration = args.get("duration_ms", 0)
            self.send_command(f"X{-speed}")
            msg = f"🤖 Turning left at {speed}% speed"
            if duration > 0:
                msg += f" for {duration}ms"
            return msg
        
        elif fn_name == "turn_right":
            speed = args.get("speed", 50)
            duration = args.get("duration_ms", 0)
            self.send_command(f"X{speed}")
            msg = f"🤖 Turning right at {speed}% speed"
            if duration > 0:
                msg += f" for {duration}ms"
            return msg
        
        elif fn_name == "stop_movement":
            self.send_command("q")
            return f"🛑 All movement stopped"
        
        elif fn_name == "move_with_differential":
            left = args.get("left_speed", 0)
            right = args.get("right_speed", 0)
            duration = args.get("duration_ms", 0)
            # Convert to forward/turn commands
            forward = (left + right) // 2
            turn = (right - left) // 2
            self.send_command(f"Y{forward}")
            self.send_command(f"X{turn}")
            msg = f"🤖 Differential drive: L={left}%, R={right}%"
            if duration > 0:
                msg += f" for {duration}ms"
            return msg
        
        # ============= BEHAVIORS =============
        elif fn_name == "scan_surroundings":
            speed = args.get("speed", "normal")
            delay_map = {"slow": 1500, "normal": 800, "fast": 400}
            delay = delay_map.get(speed, 800)
            
            # Look left, center, right
            for pos in [20, 50, 80, 50]:
                self.send_command(f"G{pos}")
            return f"👀 Scanned surroundings at {speed} speed"
        
        elif fn_name == "perform_greeting":
            # Complex greeting sequence
            self.send_command("G60")  # Look slightly right
            self.send_command("N60")  # Look up
            self.send_command("L80")  # Happy eyes
            self.send_command("R80")
            # Wave
            for pos in [70, 80, 70, 80, 70, 40]:
                self.send_command(f"B{pos}")
            return f"👋 Performed greeting sequence"
        
        elif fn_name == "navigate_to_position":
            distance = args.get("distance_cm", 0)
            angle = args.get("angle_degrees", 0)
            
            messages = []
            
            # Turn first if needed
            if angle != 0:
                turn_cmd = f"X{abs(angle)//2}" if angle > 0 else f"X{-abs(angle)//2}"
                self.send_command(turn_cmd)
                messages.append(f"Turned {angle}°")
            
            # Then move
            if distance != 0:
                # Rough conversion: assume 1cm ≈ 1% speed for 100ms (needs calibration)
                speed = min(abs(distance), 100)
                move_cmd = f"Y{speed}" if distance > 0 else f"Y{-speed}"
                self.send_command(move_cmd)
                messages.append(f"Moved {distance}cm")
            
            return f"🤖 Navigation: {', '.join(messages)}"
        
        return f"❌ Unknown robot command: {fn_name}"


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
