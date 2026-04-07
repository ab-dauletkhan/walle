# Wall-E Robot AI Control Interface

This document provides a simple interface for AI/ML developers to control the Wall-E robot.

## Hardware Overview

The Wall-E robot has the following controllable components:

### Servo Motors (7 total)
- **Head Rotation** (Channel 0): Rotate head left/right
- **Neck Top Joint** (Channel 1): Move neck up/down (top joint)
- **Neck Bottom Joint** (Channel 2): Move neck up/down (bottom joint)  
- **Right Eye** (Channel 3): Move right eye up/down
- **Left Eye** (Channel 4): Move left eye up/down
- **Left Arm** (Channel 5): Move left arm up/down
- **Right Arm** (Channel 6): Move right arm up/down

### Drive Motors (2 total)
- **Left Motor**: Controls left wheel
- **Right Motor**: Controls right wheel
- **Movement**: Both forward = move forward, opposite directions = spin

## AI Control Interface

### Basic Usage

```cpp
#include "AI_Control_Interface.hpp"

// Create AI controller instance
WallEAI robot;

void setup() {
    Serial.begin(115200);
}

void loop() {
    // Your AI logic here
    robot.setHeadRotation(75);  // Look right
    robot.driveForward(50);     // Move forward at 50% speed
    robot.wave();               // Wave hello
}
```

### Servo Control Functions

All servo functions use a **0-100 position range**:
- **0** = Minimum position (left/down)
- **50** = Center position  
- **100** = Maximum position (right/up)

#### Individual Servo Control
```cpp
// Head and neck control
robot.setHeadRotation(75);     // Look right (0=left, 50=center, 100=right)
robot.setNeckTop(60);          // Raise neck (0=down, 100=up)
robot.setNeckBottom(40);       // Lower neck bottom (0=down, 100=up)

// Eye control
robot.setRightEye(80);         // Raise right eye (0=down, 100=up)
robot.setLeftEye(80);         // Raise left eye (0=down, 100=up)

// Arm control
robot.setLeftArm(70);         // Raise left arm (0=down, 100=up)
robot.setRightArm(70);        // Raise right arm (0=down, 100=up)
```

#### Convenience Functions
```cpp
// Control both eyes together
robot.setBothEyes(60);        // Set both eyes to same position

// Control both arms together  
robot.setBothArms(50);        // Set both arms to same position

// Look in a specific direction
robot.lookAt(75, 60);         // Look right and up

// Express emotions
robot.expressEmotion(0);       // 0=happy, 1=sad, 2=surprised, 3=neutral

// Wave hello
robot.wave();

// Reset to neutral position
robot.resetToNeutral();
```

### Drive Motor Control Functions

#### Basic Movement
```cpp
// Forward/backward movement (-100 to 100)
robot.moveForward(50);         // Move forward at 50% speed
robot.moveForward(-30);       // Move backward at 30% speed

// Turning (-100 to 100)
robot.turn(-40);              // Turn left at 40% speed
robot.turn(60);               // Turn right at 60% speed

// Stop all movement
robot.stop();
```

#### Convenience Movement Functions
```cpp
// Drive forward/backward (0-100 speed)
robot.driveForward(70);       // Drive forward at 70% speed
robot.driveBackward(40);      // Drive backward at 40% speed

// Turn in place (0-100 speed)
robot.turnLeft(50);           // Turn left at 50% speed
robot.turnRight(50);          // Turn right at 50% speed
```

## Example AI Behaviors

### 1. Patrol Behavior
```cpp
void patrolBehavior() {
    // Look around
    robot.setHeadRotation(20);     // Look left
    delay(1000);
    robot.setHeadRotation(80);    // Look right
    delay(1000);
    robot.setHeadRotation(50);    // Look center
    
    // Move forward
    robot.driveForward(60);
    delay(2000);
    robot.stop();
    
    // Turn around
    robot.turnLeft(70);
    delay(1000);
    robot.stop();
}
```

### 2. Greeting Behavior
```cpp
void greetingBehavior() {
    // Look at person
    robot.lookAt(60, 70);         // Look right and up
    robot.expressEmotion(0);     // Happy expression
    
    // Wave hello
    robot.wave();
    
    // Reset to neutral
    robot.resetToNeutral();
}
```

### 3. Search Behavior
```cpp
void searchBehavior() {
    // Look in different directions
    for(int i = 0; i <= 100; i += 20) {
        robot.setHeadRotation(i);
        robot.setNeckTop(60);
        delay(500);
    }
    
    // Look up and down
    for(int i = 20; i <= 80; i += 20) {
        robot.setNeckTop(i);
        delay(500);
    }
}
```

### 4. Emotional Response
```cpp
void emotionalResponse(int detectedEmotion) {
    switch(detectedEmotion) {
        case 0: // Happy person detected
            robot.expressEmotion(0);  // Happy response
            robot.wave();
            break;
        case 1: // Sad person detected  
            robot.expressEmotion(1);  // Sad response
            robot.setBothArms(30);    // Lower arms
            break;
        case 2: // Surprised person detected
            robot.expressEmotion(2);  // Surprised response
            robot.setHeadRotation(50);
            break;
    }
}
```

## Integration with AI/ML Systems

### Python Integration Example
```python
import serial
import time

class WallEController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.serial = serial.Serial(port, baudrate)
        time.sleep(2)  # Wait for connection
    
    def send_command(self, command):
        self.serial.write(f"{command}\n".encode())
        time.sleep(0.1)  # Small delay between commands
    
    def set_head_rotation(self, position):
        self.send_command(f"G{position}")
    
    def drive_forward(self, speed):
        self.send_command(f"Y{speed}")
    
    def turn(self, turn_speed):
        self.send_command(f"X{turn_speed}")
    
    def stop(self):
        self.send_command("q")

# Usage
robot = WallEController()
robot.set_head_rotation(75)
robot.drive_forward(50)
```

### ROS Integration Example
```cpp
// ROS node for Wall-E control
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

class WallEROSController {
private:
    WallEAI robot;
    ros::Subscriber cmd_vel_sub;
    
public:
    WallEROSController() {
        cmd_vel_sub = nh.subscribe("cmd_vel", 1, &WallEROSController::cmdVelCallback, this);
    }
    
    void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg) {
        // Convert ROS velocity to Wall-E commands
        int forward = (int)(msg->linear.x * 100);
        int turn = (int)(msg->angular.z * 100);
        
        robot.moveForward(forward);
        robot.turn(turn);
    }
};
```

## Technical Notes

### Serial Communication
- **Baud Rate**: 115200
- **Format**: ASCII commands followed by newline
- **Commands**: Single character + value (e.g., "G75" for head rotation)

### Servo Calibration
The robot uses calibrated servo positions stored in the `preset` array:
```cpp
int preset[][2] = {{410,120},  // head rotation
                   {532,178},  // neck top  
                   {120,310},  // neck bottom
                   {465,271},  // eye right
                   {278,479},  // eye left
                   {340,135},  // arm left
                   {150,360}}; // arm right
```

### Motor Control
- **Speed Range**: -255 to 255 (PWM values)
- **Direction**: Positive = forward, Negative = backward
- **Differential Drive**: Different left/right speeds create turning

### Safety Considerations
- Servos automatically turn off after 6 seconds of inactivity
- Motors have acceleration/deceleration control
- All position values are constrained to valid ranges
- Emergency stop available via `robot.stop()`

## Troubleshooting

### Common Issues
1. **Servos not responding**: Check serial connection and baud rate
2. **Motors not moving**: Verify L298N connections and power supply
3. **Jittery movement**: Reduce update frequency or check power supply
4. **Commands not working**: Ensure proper serial command format

### Debug Commands
```cpp
// Check if robot is responding
Serial.println("Testing connection...");
robot.setHeadRotation(50);
delay(1000);
robot.setHeadRotation(0);
delay(1000);
robot.setHeadRotation(100);
```

This interface provides a clean, high-level API for AI systems to control the Wall-E robot without needing to understand the low-level hardware details.
