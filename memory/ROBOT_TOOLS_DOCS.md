# WALL-E Robot Control Tools Documentation

Complete set of tools for controlling the physical WALL-E robot through LLM function calling.

## Overview

The system provides **19 tools** for robot control, organized into 4 categories:

### Servo Control (7 tools)
Control servo motors for head, neck, eyes, and arms movement.

### Emotions (3 tools)
Express emotions through coordinated servo positions.

### Movement (6 tools)
Control track motors for robot locomotion.

### Behaviors (3 tools)
Complex behavioral patterns combining multiple actions.

---

## Servo Control Tools

### 1. set_head_rotation
Rotate the head left or right.

**Parameters:**
- `position` (0-100): 0=far left, 50=center, 100=far right

**Example:**
```python
# LLM calls
{"name": "set_head_rotation", "args": {"position": 75}}
# Result: Head rotated to 75% (right)
```

### 2. set_neck_position
Move the neck up or down using both joints.

**Parameters:**
- `top_position` (0-100): Upper neck joint position
- `bottom_position` (0-100): Lower neck joint position

**Example:**
```python
{"name": "set_neck_position", "args": {"top_position": 60, "bottom_position": 40}}
# Result: Neck positioned: top=60%, bottom=40%
```

### 3. set_both_eyes
Control both eyes synchronously.

**Parameters:**
- `position` (0-100): 0=down, 50=center, 100=up

**Example:**
```python
{"name": "set_both_eyes", "args": {"position": 80}}
# Result: Eyes looking up (position=80%)
```

### 4. set_individual_eyes
Control eyes independently for expressiveness.

**Parameters:**
- `left_eye` (0-100)
- `right_eye` (0-100)

**Example:**
```python
{"name": "set_individual_eyes", "args": {"left_eye": 40, "right_eye": 70}}
# Result: Eyes positioned: left=40%, right=70%
```

### 5. set_both_arms
Control both arms synchronously.

**Parameters:**
- `position` (0-100): 0=down, 50=horizontal, 100=up

**Example:**
```python
{"name": "set_both_arms", "args": {"position": 70}}
# Result: Arms raised (position=70%)
```

### 6. set_individual_arms
Control arms independently.

**Parameters:**
- `left_arm` (0-100)
- `right_arm` (0-100)

**Example:**
```python
{"name": "set_individual_arms", "args": {"left_arm": 30, "right_arm": 80}}
# Result: Arms positioned: left=30%, right=80%
```

### 7. look_at
Direct gaze to specific direction (head + neck).

**Parameters:**
- `horizontal` (0-100): 0=left, 50=straight, 100=right
- `vertical` (0-100): 0=down, 50=level, 100=up

**Example:**
```python
{"name": "look_at", "args": {"horizontal": 75, "vertical": 60}}
# Result: Looking right and up
```

---

## Emotion Tools

## Emotion Tools

### 8. express_emotion
Express an emotion using preset servo positions.

**Parameters:**
- `emotion`: "happy" | "sad" | "surprised" | "neutral" | "curious" | "confused"

**Examples:**
```python
# Happy
{"name": "express_emotion", "args": {"emotion": "happy"}}
# Result: Expressing emotion: happy
# Effect: Eyes up (80%), neck raised (60%), arms mid-position

# Sad
{"name": "express_emotion", "args": {"emotion": "sad"}}
# Result: Expressing emotion: sad
# Effect: Eyes down (20%), neck lowered (40%), arms down

# Surprised
{"name": "express_emotion", "args": {"emotion": "surprised"}}
# Effect: Eyes wide open (100%), neck extended (80%)

# Curious
{"name": "express_emotion", "args": {"emotion": "curious"}}
# Effect: Eyes slightly asymmetric (60%, 70%)

# Confused
{"name": "express_emotion", "args": {"emotion": "confused"}}
# Effect: Eyes very asymmetric (40%, 60%)
```

### 9. wave_hello
Wave hand in greeting gesture.

**Parameters:** None

**Example:**
```python
{"name": "wave_hello", "args": {}}
# Result: Waved hello!
# Effect: Right arm animation: 70->80->70->80->70->40
```

### 10. reset_to_neutral
Reset all servos to neutral position.

**Parameters:** None

**Example:**
```python
{"name": "reset_to_neutral", "args": {}}
# Result: Reset to neutral position
# Effect: Head centered, neck level, eyes straight, arms down
```

---

## Movement Tools

## Movement Tools

### 11. drive_forward
Drive forward at specified speed.

**Parameters:**
- `speed` (0-100): Speed percentage
- `duration_ms` (optional): Duration in milliseconds (0=continuous)

**Examples:**
```python
# Continuous movement
{"name": "drive_forward", "args": {"speed": 70}}
# Result: Driving forward at 70% speed

# Time-limited movement
{"name": "drive_forward", "args": {"speed": 50, "duration_ms": 2000}}
# Result: Driving forward at 50% speed for 2000ms
```

### 12. drive_backward
Drive backward.

**Parameters:**
- `speed` (0-100)
- `duration_ms` (optional)

**Example:**
```python
{"name": "drive_backward", "args": {"speed": 40, "duration_ms": 1500}}
# Result: Driving backward at 40% speed for 1500ms
```

### 13. turn_left
Turn left in place.

**Parameters:**
- `speed` (0-100): Turn rate
- `duration_ms` (optional)

**Example:**
```python
{"name": "turn_left", "args": {"speed": 60, "duration_ms": 1000}}
# Result: Turning left at 60% speed for 1000ms
```

### 14. turn_right
Turn right in place.

**Parameters:**
- `speed` (0-100)
- `duration_ms` (optional)

**Example:**
```python
{"name": "turn_right", "args": {"speed": 50}}
# Result: Turning right at 50% speed
```

### 15. stop_movement
Emergency stop for all motors.

**Parameters:** None

**Example:**
```python
{"name": "stop_movement", "args": {}}
# Result: All movement stopped
```

### 16. move_with_differential
Advanced control: set left and right track speeds independently.

**Parameters:**
- `left_speed` (-100 to 100): Negative values = backward
- `right_speed` (-100 to 100)
- `duration_ms` (optional)

**Examples:**
```python
# Arc movement to the right
{"name": "move_with_differential", "args": {"left_speed": 80, "right_speed": 40}}
# Result: Differential drive: L=80%, R=40%

# Spin left (one track forward, one backward)
{"name": "move_with_differential", "args": {"left_speed": -50, "right_speed": 50}}
```

---

## Behavior Tools

## Behavior Tools

### 17. scan_surroundings
Scan surroundings in patrol behavior.

**Parameters:**
- `speed`: "slow" | "normal" | "fast"

**Example:**
```python
{"name": "scan_surroundings", "args": {"speed": "normal"}}
# Result: Scanned surroundings at normal speed
# Effect: Head turns: left(20) -> center(50) -> right(80) -> center(50)
```

**Timing intervals:**
- slow: 1500ms between positions
- normal: 800ms between positions
- fast: 400ms between positions

### 18. perform_greeting
Execute full greeting sequence.

**Parameters:** None

**Example:**
```python
{"name": "perform_greeting", "args": {}}
# Result: Performed greeting sequence
# Effect:
#   1. Look slightly right and up
#   2. Happy eye expression (80%)
#   3. Wave hand
```

### 19. navigate_to_position
Navigate to relative position.

**Parameters:**
- `distance_cm`: Distance in centimeters (negative=backward)
- `angle_degrees` (optional): Angle to turn before moving (-180 to 180)

**Examples:**
```python
# Drive straight 50cm
{"name": "navigate_to_position", "args": {"distance_cm": 50}}
# Result: Navigation: Moved 50cm

# Turn 90 degrees right, then drive 100cm
{"name": "navigate_to_position", "args": {"distance_cm": 100, "angle_degrees": 90}}
# Result: Navigation: Turned 90 degrees, Moved 100cm

# Turn 45 degrees left, drive backward 30cm
{"name": "navigate_to_position", "args": {"distance_cm": -30, "angle_degrees": -45}}
# Result: Navigation: Turned -45 degrees, Moved -30cm
```

---

## Technical Details

## Technical Details

### RobotControlExecutor Class

**Operating modes:**

#### 1. Simulation Mode (default)
```python
executor = RobotControlExecutor()  # serial_port=None
# All commands execute virtually with console output
```

#### 2. Real Hardware Mode
```python
import serial
port = serial.Serial('/dev/ttyUSB0', 115200)
executor = RobotControlExecutor(serial_port=port)
# Commands sent to real Arduino
```

### Arduino Communication Protocol

Command format: `<LETTER><VALUE>\n`

**Servo Commands:**
- `G<0-100>` - Head rotation
- `N<0-100>` - Neck top joint
- `M<0-100>` - Neck bottom joint
- `L<0-100>` - Left eye
- `R<0-100>` - Right eye
- `A<0-100>` - Left arm
- `B<0-100>` - Right arm

**Motor Commands:**
- `Y<-100-100>` - Forward/backward speed
- `X<-100-100>` - Turn left/right speed
- `q` - Stop all motors

### Calibration

Robot uses calibration table `preset[][]` in Arduino code:
```cpp
int preset[][2] = {
    {410, 120},   // head rotation (min, max)
    {532, 178},   // neck top
    {120, 310},   // neck bottom
    {465, 271},   // eye right
    {278, 479},   // eye left
    {340, 135},   // arm left
    {150, 360}    // arm right
};
```

Positions 0-100 automatically convert to PWM values.

---

## Usage Examples

## Usage Examples

### Scenario 1: Greeting a Person
```python
# LLM receives: "Say hello to the person"
# LLM executes sequence:

1. {"name": "look_at", "args": {"horizontal": 50, "vertical": 60}}
   # Look at person

2. {"name": "express_emotion", "args": {"emotion": "happy"}}
   # Show happiness

3. {"name": "wave_hello", "args": {}}
   # Wave

4. Response to user: "Hello! Nice to see you!"
```

### Scenario 2: Searching for Object
```python
# LLM receives: "Look around for the ball"
# Sequence:

1. {"name": "express_emotion", "args": {"emotion": "curious"}}
   # Curious expression

2. {"name": "scan_surroundings", "args": {"speed": "slow"}}
   # Slow scan

3. {"name": "turn_left", "args": {"speed": 30, "duration_ms": 2000}}
   # Turn to expand view

4. {"name": "scan_surroundings", "args": {"speed": "slow"}}
   # Scan again

5. Response: "I'm scanning the area for the ball..."
```

### Scenario 3: Navigation to Point
```python
# LLM receives: "Move 1 meter forward and turn right 90 degrees"

1. {"name": "drive_forward", "args": {"speed": 60, "duration_ms": 3000}}
   # Approximately 1 meter forward

2. {"name": "stop_movement", "args": {}}
   # Stop

3. {"name": "turn_right", "args": {"speed": 50, "duration_ms": 1500}}
   # Turn approximately 90 degrees

4. {"name": "stop_movement", "args": {}}

5. Response: "Done! I've moved forward and turned right."
```

### Scenario 4: Emotional Response
```python
# LLM receives: "The user looks sad"

1. {"name": "express_emotion", "args": {"emotion": "sad"}}
   # Show empathy

2. {"name": "look_at", "args": {"horizontal": 50, "vertical": 55}}
   # Eye contact

3. Response: "I'm sorry you're feeling down. Is there anything I can do to help?"
```

---

## Important Notes

## Important Notes

### Safety
1. **Servo timeouts**: Servos automatically disable after 6 seconds of inactivity
2. **Emergency stop**: Always available via `stop_movement()`
3. **Value constraints**: All values automatically constrained to valid ranges

### Performance
1. **Delays**: Add small delays between commands for smooth operation
2. **Power**: Ensure power supply can handle all motors
3. **Sequencing**: Avoid overloading robot with too many simultaneous commands

### Calibration
1. Use `wall-e_calibration.ino` to configure servo ranges
2. Update `preset[][]` table after calibration
3. Test all positions before real usage

---

## Integration with walle_enhanced.py

Tools are automatically integrated into the system:

```python
# In walle_enhanced.py
from robot_tools import get_robot_control_tools, RobotControlExecutor, get_robot_tool_names

# Initialize
robot_controller = RobotControlExecutor()  # Simulation mode

# Get all tools
all_tools = get_robot_control_tools() + get_memory_tools() + get_personality_tools()

# Execute commands
if fn_name in get_robot_tool_names():
    result = robot_controller.execute(fn_name, args)
```

The LLM automatically has access to all 19 tools and can use them naturally in conversation.

---

## How It Works: LLM to Robot Communication

When the LLM decides to control the robot, here's the complete flow:

### Step 1: LLM Decision
```
User: "Wave hello to me"
LLM thinks: "I need to use wave_hello function"
```

### Step 2: Function Call
```python
# LLM generates function call
{
    "name": "wave_hello",
    "args": {}
}
```

### Step 3: Python Execution
```python
# walle_enhanced.py receives call
fn_name = "wave_hello"
args = {}

# Routes to robot executor
result = robot_controller.execute(fn_name, args)
```

### Step 4: Serial Commands
```python
# RobotControlExecutor.execute() translates to serial commands
def execute(self, fn_name, args):
    if fn_name == "wave_hello":
        # Animation sequence
        for pos in [70, 80, 70, 80, 70, 40]:
            self.send_command(f"B{pos}")  # Right arm
```

### Step 5: Arduino Receives
```
Serial Monitor receives:
B70
B80
B70
B80
B70
B40
```

### Step 6: Arduino Executes
```cpp
// In wall-e.ino
void loop() {
    if (Serial.available()) {
        char cmd = Serial.read();
        int value = Serial.parseInt();
        
        if (cmd == 'B') {  // Right arm command
            // Convert 0-100 to PWM (150-360 from calibration)
            int pwm = map(value, 0, 100, preset[6][0], preset[6][1]);
            pwm.setPWM(6, 0, pwm);  // Channel 6 = Right arm
        }
    }
}
```

### Step 7: Physical Movement
```
Arduino sends PWM signal to servo driver (PCA9685)
-> Servo driver sends signal to right arm servo
-> Servo motor rotates to position
-> Arm waves!
```

### Complete Data Flow Diagram

```
USER INPUT
    |
    v
LLM (walle_enhanced.py)
    |
    | decides: "I should wave"
    v
FUNCTION CALL: wave_hello()
    |
    v
robot_controller.execute("wave_hello", {})
    |
    | translates to serial commands
    v
SERIAL PORT: "B70\n", "B80\n", ...
    |
    | USB cable
    v
ARDUINO (wall-e.ino)
    |
    | parses: cmd='B', value=70
    | maps: 70 -> PWM value (based on calibration)
    v
PCA9685 SERVO DRIVER
    |
    | PWM signal on channel 6
    v
SERVO MOTOR (Right Arm)
    |
    | mechanical rotation
    v
PHYSICAL MOVEMENT: Arm waves!
```

### Example: drive_forward(50)

```
1. LLM: {"name": "drive_forward", "args": {"speed": 50}}
   
2. Python: robot_controller.execute("drive_forward", {"speed": 50})
   
3. Serial: "Y50\n"  (Y = forward/backward command)
   
4. Arduino: 
   - Reads 'Y' and value 50
   - Converts to motor speed: map(50, 0, 100, 0, 255) = 127
   - Sets both L298N motors: 
     * Left motor: digitalWrite(IN1, HIGH), analogWrite(ENA, 127)
     * Right motor: digitalWrite(IN3, HIGH), analogWrite(ENB, 127)
   
5. L298N Motor Driver:
   - Sends power to left track motor
   - Sends power to right track motor
   
6. DC Motors: Both tracks rotate forward at 50% speed
   
7. Robot moves forward!
```

### Key Points

1. **Abstraction Layers**: LLM doesn't need to know PWM values, motor drivers, or hardware details
2. **Serial Protocol**: Simple ASCII commands (G, N, M, L, R, A, B, Y, X, q)
3. **Calibration**: `preset[][]` table maps 0-100 positions to actual servo ranges
4. **Simulation Mode**: Without serial connection, commands just print to console
5. **Real Mode**: With serial connection, commands control actual hardware

This architecture allows the LLM to naturally control a physical robot using high-level function calls, while all the low-level hardware control is handled transparently.

---

## Additional Information

- **Arduino code**: `/wall-e/wall-e.ino`
- **Calibration**: `/wall-e/wall-e_calibration.ino`
- **Arduino documentation**: `/wall-e/AI_Control_Documentation.md`
- **Usage examples**: `/wall-e/AI_Example_Usage.ino`

---

*Documentation for WALL-E Robot Project*  
*Version 1.0 - November 2025*
