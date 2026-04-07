# Wall-E AI Direct Motor Control Interface

This document provides direct motor control functions for AI/ML developers to control the Wall-E robot by specifying exact degrees of rotation for each motor.

## Overview

The AI can control each motor individually by specifying degrees of rotation:
- **Positive degrees** = one direction
- **Negative degrees** = opposite direction  
- **360 degrees** = full rotation
- **-360 degrees** = full rotation in opposite direction

## Motor Control Functions

### Servo Motors (7 total)

#### Head and Neck Control
```cpp
// Rotate head left/right by degrees
robot.rotateHead(45);          // Turn head right 45 degrees
robot.rotateHead(-30);         // Turn head left 30 degrees
robot.rotateHead(360);         // Full head rotation

// Move neck top joint up/down by degrees
robot.moveNeckTop(20);         // Move neck up 20 degrees
robot.moveNeckTop(-15);        // Move neck down 15 degrees

// Move neck bottom joint up/down by degrees
robot.moveNeckBottom(25);      // Move neck bottom up 25 degrees
robot.moveNeckBottom(-20);     // Move neck bottom down 20 degrees
```

#### Eye Control
```cpp
// Move right eye up/down by degrees
robot.moveRightEye(30);        // Move right eye up 30 degrees
robot.moveRightEye(-25);       // Move right eye down 25 degrees

// Move left eye up/down by degrees
robot.moveLeftEye(30);         // Move left eye up 30 degrees
robot.moveLeftEye(-25);        // Move left eye down 25 degrees

// Move both eyes together
robot.moveBothEyes(20);        // Move both eyes up 20 degrees
robot.moveBothEyes(-15);       // Move both eyes down 15 degrees
```

#### Arm Control
```cpp
// Move left arm up/down by degrees
robot.moveLeftArm(60);         // Move left arm up 60 degrees
robot.moveLeftArm(-45);        // Move left arm down 45 degrees

// Move right arm up/down by degrees
robot.moveRightArm(60);        // Move right arm up 60 degrees
robot.moveRightArm(-45);       // Move right arm down 45 degrees

// Move both arms together
robot.moveBothArms(50);        // Move both arms up 50 degrees
robot.moveBothArms(-30);       // Move both arms down 30 degrees
```

### Drive Motors (2 total)

#### Individual Wheel Control
```cpp
// Rotate left wheel by degrees
robot.rotateLeftWheel(360);    // Rotate left wheel 1 full turn forward
robot.rotateLeftWheel(-180);   // Rotate left wheel 180 degrees backward

// Rotate right wheel by degrees
robot.rotateRightWheel(360);   // Rotate right wheel 1 full turn forward
robot.rotateRightWheel(-180);  // Rotate right wheel 180 degrees backward

// Move both wheels together
robot.moveBothWheels(720);     // Move both wheels 2 full turns forward
robot.moveBothWheels(-360);    // Move both wheels 1 full turn backward
```

#### Movement Functions
```cpp
// Move robot forward/backward by degrees
robot.moveForward(360);        // Move forward 1 full rotation
robot.moveBackward(180);       // Move backward 180 degrees

// Turn robot left/right by degrees
robot.turnLeft(90);            // Turn left 90 degrees
robot.turnRight(90);           // Turn right 90 degrees

// Spin robot in place by degrees
robot.spinRobot(360);          // Spin 360 degrees right
robot.spinRobot(-180);         // Spin 180 degrees left

// Stop all motors
robot.stopAll();              // Stop all motors immediately
```

## AI Usage Examples

### 1. Basic AI Movement
```cpp
// AI decides to look around
robot.rotateHead(-60);         // Look left 60 degrees
delay(1000);
robot.rotateHead(120);         // Look right 120 degrees
delay(1000);
robot.rotateHead(-60);         // Look center

// AI decides to move forward
robot.moveForward(360);        // Move forward 1 rotation
delay(2000);

// AI decides to turn
robot.turnLeft(90);            // Turn left 90 degrees
delay(1000);
```

### 2. AI Object Interaction
```cpp
// AI detects object and approaches
robot.rotateHead(30);          // Look at object
robot.moveNeckTop(15);         // Look up at object
robot.moveForward(180);        // Approach object

// AI waves at object
robot.moveLeftArm(60);         // Left arm up
robot.moveRightArm(60);        // Right arm up
delay(1000);
robot.moveLeftArm(-60);        // Arms down
robot.moveRightArm(-60);
```

### 3. AI Navigation
```cpp
// AI navigation decision making
if (obstacleDetected) {
    robot.turnRight(90);       // Turn right 90 degrees
    robot.moveForward(180);     // Move forward
    robot.turnLeft(90);         // Turn back left
}
else if (targetDetected) {
    robot.moveForward(360);     // Move toward target
    robot.rotateHead(0);        // Look at target
}
else {
    robot.moveForward(180);     // Continue forward
}
```

### 4. AI Emotional Expression
```cpp
// AI expresses happiness
robot.moveBothEyes(20);        // Eyes up (happy)
robot.moveNeckTop(10);         // Head up slightly
robot.moveBothArms(45);        // Arms up (wave)

// AI expresses sadness
robot.moveBothEyes(-15);       // Eyes down (sad)
robot.moveNeckTop(-10);        // Head down
robot.moveBothArms(-30);       // Arms down
```

## Advanced AI Behaviors

### Patrol Behavior
```cpp
void aiPatrol() {
    // Look around
    robot.rotateHead(-45);     // Look left
    delay(1000);
    robot.rotateHead(90);      // Look right
    delay(1000);
    robot.rotateHead(-45);     // Look center
    
    // Move forward
    robot.moveForward(360);    // Move forward 1 rotation
    delay(2000);
    
    // Turn around
    robot.turnLeft(180);       // Turn around
    delay(1000);
}
```

### Search Behavior
```cpp
void aiSearch() {
    // Horizontal search
    for(int i = -60; i <= 60; i += 30) {
        robot.rotateHead(i);
        delay(500);
    }
    
    // Vertical search
    for(int i = -30; i <= 30; i += 15) {
        robot.moveNeckTop(i);
        delay(500);
    }
    
    // Eye movement
    robot.moveBothEyes(20);    // Look up
    delay(1000);
    robot.moveBothEyes(-20);   // Look down
    delay(1000);
}
```

### Greeting Behavior
```cpp
void aiGreeting() {
    // Look at person
    robot.rotateHead(0);        // Look forward
    robot.moveNeckTop(15);     // Look up slightly
    
    // Wave
    robot.moveLeftArm(60);     // Left arm up
    robot.moveRightArm(60);    // Right arm up
    delay(1000);
    robot.moveLeftArm(-60);    // Arms down
    robot.moveRightArm(-60);
}
```

## Technical Implementation

### Serial Command Format
The interface sends commands in the format:
```
MOTOR_NAME:degrees
```

Examples:
- `HEAD_ROTATE:45` - Rotate head 45 degrees
- `LEFT_WHEEL:360` - Rotate left wheel 360 degrees
- `STOP_ALL` - Stop all motors
- `RESET_CENTER` - Reset all servos to center

### Motor Mapping
- **Servo Motors**: Channels 0-6 on PCA9685
- **Drive Motors**: L298N motor controller
- **Degrees**: Converted to appropriate PWM/speed values

### Safety Features
- All degree values are processed by the main controller
- Motors have built-in acceleration/deceleration
- Emergency stop available via `robot.stopAll()`
- Servos automatically turn off after inactivity

## Integration Examples

### Python Integration
```python
import serial
import time

class WallEController:
    def __init__(self, port='/dev/ttyUSB0'):
        self.serial = serial.Serial(port, 115200)
    
    def rotate_head(self, degrees):
        self.serial.write(f"HEAD_ROTATE:{degrees}\n".encode())
    
    def move_forward(self, degrees):
        self.serial.write(f"LEFT_WHEEL:{degrees}\n".encode())
        self.serial.write(f"RIGHT_WHEEL:{degrees}\n".encode())

# AI usage
robot = WallEController()
robot.rotate_head(45)      # Look right 45 degrees
robot.move_forward(360)    # Move forward 1 rotation
```

### ROS Integration
```cpp
// ROS node for Wall-E control
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

class WallEROSController {
private:
    WallEAI robot;
    ros::Subscriber cmd_vel_sub;
    
public:
    void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg) {
        // Convert ROS velocity to degrees
        float left_degrees = msg->linear.x * 100;
        float right_degrees = msg->linear.x * 100;
        float turn_degrees = msg->angular.z * 100;
        
        robot.rotateLeftWheel(left_degrees - turn_degrees);
        robot.rotateRightWheel(right_degrees + turn_degrees);
    }
};
```

This interface provides direct, degree-based control that allows AI systems to make precise motor control decisions without needing to understand the underlying hardware implementation.
