/**
 * WALL-E AI CONTROL INTERFACE
 * 
 * Direct motor control interface for AI/ML developers
 * 
 * This header provides direct control functions for:
 * - 7 servo motors (head, neck, eyes, arms) - control by degrees
 * - 2 drive motors (left/right wheels) - control by degrees
 * 
 * AI can specify exact degrees of rotation for each motor
 */

#ifndef AI_CONTROL_INTERFACE_HPP
#define AI_CONTROL_INTERFACE_HPP

#include <Arduino.h>

class WallEAI {
private:
    HardwareSerial* serialPort;
    
public:
    // Constructor - pass Serial or Serial1, etc.
    WallEAI(HardwareSerial* serial = &Serial) : serialPort(serial) {}
    
    // ========================================
    // SERVO MOTOR CONTROL FUNCTIONS (by degrees)
    // ========================================
    
    /**
     * Rotate head left/right by degrees
     * @param degrees Positive = right, Negative = left, 360 = full rotation
     */
    void rotateHead(float degrees) {
        serialPort->print("HEAD_ROTATE:");
        serialPort->println(degrees);
    }
    
    /**
     * Move neck top joint up/down by degrees
     * @param degrees Positive = up, Negative = down
     */
    void moveNeckTop(float degrees) {
        serialPort->print("NECK_TOP:");
        serialPort->println(degrees);
    }
    
    /**
     * Move neck bottom joint up/down by degrees
     * @param degrees Positive = up, Negative = down
     */
    void moveNeckBottom(float degrees) {
        serialPort->print("NECK_BOTTOM:");
        serialPort->println(degrees);
    }
    
    /**
     * Move right eye up/down by degrees
     * @param degrees Positive = up, Negative = down
     */
    void moveRightEye(float degrees) {
        serialPort->print("RIGHT_EYE:");
        serialPort->println(degrees);
    }
    
    /**
     * Move left eye up/down by degrees
     * @param degrees Positive = up, Negative = down
     */
    void moveLeftEye(float degrees) {
        serialPort->print("LEFT_EYE:");
        serialPort->println(degrees);
    }
    
    /**
     * Move left arm up/down by degrees
     * @param degrees Positive = up, Negative = down
     */
    void moveLeftArm(float degrees) {
        serialPort->print("LEFT_ARM:");
        serialPort->println(degrees);
    }
    
    /**
     * Move right arm up/down by degrees
     * @param degrees Positive = up, Negative = down
     */
    void moveRightArm(float degrees) {
        serialPort->print("RIGHT_ARM:");
        serialPort->println(degrees);
    }
    
    // ========================================
    // DRIVE MOTOR CONTROL FUNCTIONS (by degrees)
    // ========================================
    
    /**
     * Rotate left wheel by degrees
     * @param degrees Positive = forward, Negative = backward, 360 = full rotation
     */
    void rotateLeftWheel(float degrees) {
        serialPort->print("LEFT_WHEEL:");
        serialPort->println(degrees);
    }
    
    /**
     * Rotate right wheel by degrees
     * @param degrees Positive = forward, Negative = backward, 360 = full rotation
     */
    void rotateRightWheel(float degrees) {
        serialPort->print("RIGHT_WHEEL:");
        serialPort->println(degrees);
    }
    
    /**
     * Stop all motors
     */
    void stopAll() {
        serialPort->println("STOP_ALL");
    }
    
    // ========================================
    // CONVENIENCE FUNCTIONS
    // ========================================
    
    /**
     * Move both eyes by same amount
     * @param degrees Positive = up, Negative = down
     */
    void moveBothEyes(float degrees) {
        moveRightEye(degrees);
        moveLeftEye(degrees);
    }
    
    /**
     * Move both arms by same amount
     * @param degrees Positive = up, Negative = down
     */
    void moveBothArms(float degrees) {
        moveLeftArm(degrees);
        moveRightArm(degrees);
    }
    
    /**
     * Move both wheels by same amount (forward/backward)
     * @param degrees Positive = forward, Negative = backward
     */
    void moveBothWheels(float degrees) {
        rotateLeftWheel(degrees);
        rotateRightWheel(degrees);
    }
    
    /**
     * Turn robot by rotating wheels in opposite directions
     * @param degrees Positive = turn right, Negative = turn left
     */
    void turnRobot(float degrees) {
        rotateLeftWheel(-degrees);
        rotateRightWheel(degrees);
    }
    
    /**
     * Move robot forward by rotating both wheels
     * @param degrees How far to move forward
     */
    void moveForward(float degrees) {
        moveBothWheels(degrees);
    }
    
    /**
     * Move robot backward by rotating both wheels
     * @param degrees How far to move backward
     */
    void moveBackward(float degrees) {
        moveBothWheels(-degrees);
    }
    
    /**
     * Turn left by rotating wheels in opposite directions
     * @param degrees How much to turn left
     */
    void turnLeft(float degrees) {
        turnRobot(-degrees);
    }
    
    /**
     * Turn right by rotating wheels in opposite directions
     * @param degrees How much to turn right
     */
    void turnRight(float degrees) {
        turnRobot(degrees);
    }
    
    /**
     * Spin robot in place (wheels rotate in opposite directions)
     * @param degrees Positive = spin right, Negative = spin left
     */
    void spinRobot(float degrees) {
        turnRobot(degrees);
    }
    
    /**
     * Reset all servos to center positions
     */
    void resetToCenter() {
        serialPort->println("RESET_CENTER");
    }

#endif /* AI_CONTROL_INTERFACE_HPP */
