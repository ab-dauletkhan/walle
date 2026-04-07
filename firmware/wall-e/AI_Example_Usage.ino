/**
 * WALL-E AI CONTROL EXAMPLE
 * 
 * This example demonstrates how to use the AI Control Interface
 * for autonomous robot behaviors.
 * 
 * Upload this sketch to test the AI interface functions.
 */

#include "AI_Control_Interface.hpp"

// Create AI controller instance
WallEAI robot;

void setup() {
    Serial.begin(115200);
    Serial.println("Wall-E AI Control Example Starting...");
    
    // Wait for serial connection
    while (!Serial) {
        delay(10);
    }
    
    Serial.println("AI Control Interface Ready!");
    Serial.println("Available commands:");
    Serial.println("- Press '1' for Patrol Behavior");
    Serial.println("- Press '2' for Greeting Behavior"); 
    Serial.println("- Press '3' for Search Behavior");
    Serial.println("- Press '4' for Emotional Demo");
    Serial.println("- Press '5' for Movement Demo");
    Serial.println("- Press 'r' to Reset to Neutral");
}

void loop() {
    if (Serial.available()) {
        char command = Serial.read();
        
        switch(command) {
            case '1':
                Serial.println("Starting Patrol Behavior...");
                patrolBehavior();
                break;
                
            case '2':
                Serial.println("Starting Greeting Behavior...");
                greetingBehavior();
                break;
                
            case '3':
                Serial.println("Starting Search Behavior...");
                searchBehavior();
                break;
                
            case '4':
                Serial.println("Starting Emotional Demo...");
                emotionalDemo();
                break;
                
            case '5':
                Serial.println("Starting Movement Demo...");
                movementDemo();
                break;
                
            case 'r':
                Serial.println("Resetting to neutral...");
                robot.resetToNeutral();
                robot.stop();
                break;
                
            default:
                Serial.println("Unknown command. Press 1-5 or 'r'");
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
 * Patrol Behavior - Robot looks around and moves
 */
void patrolBehavior() {
    Serial.println("Patrol: Looking around...");
    
    // Look in different directions
    robot.setHeadRotation(20);     // Look left
    delay(1000);
    robot.setHeadRotation(80);     // Look right  
    delay(1000);
    robot.setHeadRotation(50);    // Look center
    
    // Look up and down
    robot.setNeckTop(70);          // Look up
    delay(1000);
    robot.setNeckTop(30);          // Look down
    delay(1000);
    robot.setNeckTop(50);          // Look center
    
    Serial.println("Patrol: Moving forward...");
    robot.driveForward(60);       // Move forward
    delay(2000);
    robot.stop();
    
    Serial.println("Patrol: Turning around...");
    robot.turnLeft(70);           // Turn left
    delay(1500);
    robot.stop();
    
    Serial.println("Patrol: Moving back...");
    robot.driveBackward(50);      // Move backward
    delay(2000);
    robot.stop();
    
    Serial.println("Patrol complete!");
}

/**
 * Greeting Behavior - Robot greets someone
 */
void greetingBehavior() {
    Serial.println("Greeting: Looking at person...");
    
    // Look at person (right and up)
    robot.lookAt(60, 70);
    delay(1000);
    
    // Express happiness
    robot.expressEmotion(0);      // Happy expression
    delay(1000);
    
    // Wave hello
    Serial.println("Greeting: Waving...");
    robot.wave();
    delay(2000);
    
    // Reset to neutral
    robot.resetToNeutral();
    
    Serial.println("Greeting complete!");
}

/**
 * Search Behavior - Robot searches for something
 */
void searchBehavior() {
    Serial.println("Search: Scanning area...");
    
    // Look in different horizontal directions
    for(int i = 0; i <= 100; i += 25) {
        robot.setHeadRotation(i);
        delay(800);
    }
    
    // Look in different vertical directions
    for(int i = 20; i <= 80; i += 20) {
        robot.setNeckTop(i);
        delay(800);
    }
    
    // Look with eyes
    robot.setBothEyes(20);        // Look down
    delay(1000);
    robot.setBothEyes(80);        // Look up
    delay(1000);
    robot.setBothEyes(50);        // Look center
    
    // Reset to neutral
    robot.resetToNeutral();
    
    Serial.println("Search complete!");
}

/**
 * Emotional Demo - Shows different emotional expressions
 */
void emotionalDemo() {
    Serial.println("Emotional Demo: Happy...");
    robot.expressEmotion(0);      // Happy
    delay(2000);
    
    Serial.println("Emotional Demo: Sad...");
    robot.expressEmotion(1);      // Sad
    delay(2000);
    
    Serial.println("Emotional Demo: Surprised...");
    robot.expressEmotion(2);      // Surprised
    delay(2000);
    
    Serial.println("Emotional Demo: Neutral...");
    robot.expressEmotion(3);      // Neutral
    delay(2000);
    
    Serial.println("Emotional demo complete!");
}

/**
 * Movement Demo - Shows different movement patterns
 */
void movementDemo() {
    Serial.println("Movement: Forward...");
    robot.driveForward(70);
    delay(2000);
    robot.stop();
    delay(500);
    
    Serial.println("Movement: Backward...");
    robot.driveBackward(50);
    delay(2000);
    robot.stop();
    delay(500);
    
    Serial.println("Movement: Turn left...");
    robot.turnLeft(60);
    delay(1500);
    robot.stop();
    delay(500);
    
    Serial.println("Movement: Turn right...");
    robot.turnRight(60);
    delay(1500);
    robot.stop();
    delay(500);
    
    Serial.println("Movement: Spin...");
    robot.turnLeft(80);
    delay(2000);
    robot.stop();
    
    Serial.println("Movement demo complete!");
}

/**
 * AI Decision Making Example
 * This shows how an AI system might make decisions
 */
void aiDecisionExample() {
    // Simulate AI decision making
    int detectedObject = random(0, 4);  // 0=nothing, 1=person, 2=obstacle, 3=target
    
    switch(detectedObject) {
        case 0: // Nothing detected
            Serial.println("AI: Nothing detected, continuing patrol...");
            robot.driveForward(40);
            delay(1000);
            robot.stop();
            break;
            
        case 1: // Person detected
            Serial.println("AI: Person detected, greeting...");
            robot.lookAt(50, 60);
            robot.expressEmotion(0);
            robot.wave();
            delay(2000);
            robot.resetToNeutral();
            break;
            
        case 2: // Obstacle detected
            Serial.println("AI: Obstacle detected, avoiding...");
            robot.turnRight(60);
            delay(1000);
            robot.stop();
            robot.driveForward(50);
            delay(1500);
            robot.stop();
            break;
            
        case 3: // Target detected
            Serial.println("AI: Target detected, approaching...");
            robot.lookAt(50, 50);
            robot.driveForward(60);
            delay(2000);
            robot.stop();
            robot.expressEmotion(2);  // Surprised
            break;
    }
}
