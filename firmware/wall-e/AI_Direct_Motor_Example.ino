/**
 * WALL-E AI DIRECT MOTOR CONTROL EXAMPLE
 * 
 * This example shows how AI can directly control each motor by degrees
 * 
 * AI can specify exact degrees of rotation for each motor:
 * - Positive degrees = one direction
 * - Negative degrees = opposite direction  
 * - 360 degrees = full rotation
 */

#include "AI_Control_Interface.hpp"

// Create AI controller instance
WallEAI robot;

void setup() {
    Serial.begin(115200);
    Serial.println("Wall-E Direct Motor Control Example");
    Serial.println("AI can control each motor individually by degrees");
    
    // Wait for serial connection
    while (!Serial) {
        delay(10);
    }
    
    Serial.println("Available AI motor control functions:");
    Serial.println("- rotateHead(degrees)");
    Serial.println("- moveNeckTop(degrees)");
    Serial.println("- moveNeckBottom(degrees)");
    Serial.println("- moveRightEye(degrees)");
    Serial.println("- moveLeftEye(degrees)");
    Serial.println("- moveLeftArm(degrees)");
    Serial.println("- moveRightArm(degrees)");
    Serial.println("- rotateLeftWheel(degrees)");
    Serial.println("- rotateRightWheel(degrees)");
    Serial.println();
    Serial.println("Press '1' for Servo Demo");
    Serial.println("Press '2' for Wheel Demo");
    Serial.println("Press '3' for AI Movement Demo");
    Serial.println("Press 'r' to Reset");
}

void loop() {
    if (Serial.available()) {
        char command = Serial.read();
        
        switch(command) {
            case '1':
                servoDemo();
                break;
            case '2':
                wheelDemo();
                break;
            case '3':
                aiMovementDemo();
                break;
            case 'r':
                robot.resetToCenter();
                robot.stopAll();
                Serial.println("Reset to center");
                break;
            default:
                Serial.println("Press 1, 2, 3, or 'r'");
                break;
        }
        
        // Clear any remaining serial data
        while (Serial.available()) {
            Serial.read();
        }
    }
    
    delay(100);
}

/**
 * Demo: AI controlling servo motors by degrees
 */
void servoDemo() {
    Serial.println("=== SERVO DEMO ===");
    
    // AI decides to look around
    Serial.println("AI: Looking left 45 degrees");
    robot.rotateHead(-45);
    delay(2000);
    
    Serial.println("AI: Looking right 90 degrees");
    robot.rotateHead(90);
    delay(2000);
    
    Serial.println("AI: Looking up 30 degrees");
    robot.moveNeckTop(30);
    delay(2000);
    
    Serial.println("AI: Looking down 20 degrees");
    robot.moveNeckTop(-20);
    delay(2000);
    
    Serial.println("AI: Moving eyes up 15 degrees");
    robot.moveBothEyes(15);
    delay(2000);
    
    Serial.println("AI: Moving arms up 45 degrees");
    robot.moveBothArms(45);
    delay(2000);
    
    Serial.println("AI: Moving arms down 30 degrees");
    robot.moveBothArms(-30);
    delay(2000);
    
    Serial.println("Servo demo complete!");
}

/**
 * Demo: AI controlling wheel motors by degrees
 */
void wheelDemo() {
    Serial.println("=== WHEEL DEMO ===");
    
    // AI decides to move forward
    Serial.println("AI: Moving forward 720 degrees (2 full rotations)");
    robot.moveForward(720);
    delay(3000);
    
    // AI decides to turn left
    Serial.println("AI: Turning left 180 degrees");
    robot.turnLeft(180);
    delay(2000);
    
    // AI decides to move backward
    Serial.println("AI: Moving backward 360 degrees (1 full rotation)");
    robot.moveBackward(360);
    delay(2000);
    
    // AI decides to spin
    Serial.println("AI: Spinning right 360 degrees");
    robot.spinRobot(360);
    delay(2000);
    
    // AI decides to stop
    Serial.println("AI: Stopping all motors");
    robot.stopAll();
    
    Serial.println("Wheel demo complete!");
}

/**
 * Demo: AI making complex movement decisions
 */
void aiMovementDemo() {
    Serial.println("=== AI MOVEMENT DEMO ===");
    
    // AI decision: Look around first
    Serial.println("AI: Scanning environment...");
    robot.rotateHead(-60);        // Look left 60 degrees
    delay(1000);
    robot.rotateHead(120);         // Look right 120 degrees
    delay(1000);
    robot.rotateHead(-60);        // Look center
    delay(1000);
    
    // AI decision: Move forward while looking up
    Serial.println("AI: Moving forward while looking up...");
    robot.moveForward(360);        // Move forward 1 rotation
    robot.moveNeckTop(20);        // Look up 20 degrees
    delay(2000);
    
    // AI decision: Turn and look at something
    Serial.println("AI: Turning to look at object...");
    robot.turnRight(90);           // Turn right 90 degrees
    robot.rotateHead(30);          // Look right 30 degrees
    robot.moveNeckTop(15);         // Look up 15 degrees
    delay(2000);
    
    // AI decision: Approach object
    Serial.println("AI: Approaching object...");
    robot.moveForward(180);        // Move forward half rotation
    delay(1500);
    
    // AI decision: Wave at object
    Serial.println("AI: Waving at object...");
    robot.moveLeftArm(60);        // Left arm up 60 degrees
    robot.moveRightArm(60);       // Right arm up 60 degrees
    delay(1000);
    robot.moveLeftArm(-30);       // Left arm down 30 degrees
    robot.moveRightArm(-30);      // Right arm down 30 degrees
    delay(1000);
    
    // AI decision: Back away
    Serial.println("AI: Backing away...");
    robot.moveBackward(270);      // Move backward 270 degrees
    delay(2000);
    
    // AI decision: Turn around and leave
    Serial.println("AI: Turning around and leaving...");
    robot.turnLeft(180);          // Turn around 180 degrees
    robot.moveForward(720);       // Move forward 2 rotations
    delay(3000);
    
    robot.stopAll();
    Serial.println("AI movement demo complete!");
}

/**
 * Example: AI making autonomous decisions
 */
void aiAutonomousExample() {
    // Simulate AI sensor input
    int obstacleDetected = random(0, 2);
    int personDetected = random(0, 2);
    int targetDetected = random(0, 2);
    
    if (obstacleDetected) {
        Serial.println("AI: Obstacle detected, avoiding...");
        robot.turnRight(90);           // Turn right 90 degrees
        robot.moveForward(180);       // Move forward 180 degrees
        robot.turnLeft(90);           // Turn back left 90 degrees
    }
    else if (personDetected) {
        Serial.println("AI: Person detected, greeting...");
        robot.rotateHead(0);          // Look at person
        robot.moveNeckTop(10);        // Nod slightly
        robot.moveBothArms(45);       // Wave
        delay(2000);
        robot.moveBothArms(-45);      // Lower arms
    }
    else if (targetDetected) {
        Serial.println("AI: Target detected, approaching...");
        robot.moveForward(360);       // Move forward 1 rotation
        robot.moveNeckTop(20);        // Look up at target
    }
    else {
        Serial.println("AI: No targets, continuing patrol...");
        robot.moveForward(180);    // Move forward half rotation
        robot.rotateHead(-30);       // Look left 30 degrees
        delay(1000);
        robot.rotateHead(60);        // Look right 60 degrees
        delay(1000);
        robot.rotateHead(-30);       // Look center
    }
}
