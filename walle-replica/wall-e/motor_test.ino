#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// L298N pins (aligned with working sketch)
const int ENA = 3;   // Enable Motor A (Left)
const int IN1 = 2;   // Motor A input 1
const int IN2 = 4;   // Motor A input 2
const int IN3 = 5;   // Motor B input 1 (Right)
const int IN4 = 7;   // Motor B input 2 (Right)
const int ENB = 6;   // Enable Motor B (Right)
const int OE  = 11;  // PCA9685 Output Enable (LOW = enabled)

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo pulse range (ticks) for PCA9685 at 50-60Hz; adjust if needed
#define SERVOMIN 150
#define SERVOMAX 600

int mapAngleToTicks(int angle) {
	return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

void sweepServo(int channel, const __FlashStringHelper* label) {
	Serial.print(F("Servo test: ")); Serial.println(label);
	for (int pos = 0; pos <= 180; pos += 15) {
		pwm.setPWM(channel, 0, mapAngleToTicks(pos));
		delay(200);
	}
	for (int pos = 180; pos >= 0; pos -= 15) {
		pwm.setPWM(channel, 0, mapAngleToTicks(pos));
		delay(200);
	}
	// center
	pwm.setPWM(channel, 0, mapAngleToTicks(90));
	delay(300);
}

void setMotorA(int pwmValue) {
	pwmValue = constrain(pwmValue, -255, 255);
	if (pwmValue > 0) {
		digitalWrite(IN1, HIGH);
		digitalWrite(IN2, LOW);
		analogWrite(ENA, pwmValue);
	} else if (pwmValue < 0) {
		digitalWrite(IN1, LOW);
		digitalWrite(IN2, HIGH);
		analogWrite(ENA, -pwmValue);
	} else {
		digitalWrite(IN1, LOW);
		digitalWrite(IN2, LOW);
		analogWrite(ENA, 0);
	}
}

void setMotorB(int pwmValue) {
	pwmValue = constrain(pwmValue, -255, 255);
	if (pwmValue > 0) {
		digitalWrite(IN3, HIGH);
		digitalWrite(IN4, LOW);
		analogWrite(ENB, pwmValue);
	} else if (pwmValue < 0) {
		digitalWrite(IN3, LOW);
		digitalWrite(IN4, HIGH);
		analogWrite(ENB, -pwmValue);
	} else {
		digitalWrite(IN3, LOW);
		digitalWrite(IN4, LOW);
		analogWrite(ENB, 0);
	}
}

void setup() {
	Serial.begin(115200);
	
	pinMode(ENA, OUTPUT);
	pinMode(IN1, OUTPUT);
	pinMode(IN2, OUTPUT);
	pinMode(IN3, OUTPUT);
	pinMode(IN4, OUTPUT);
	pinMode(ENB, OUTPUT);
	pinMode(OE, OUTPUT);

	// Enable PCA9685 outputs (LOW = enabled)
	digitalWrite(OE, LOW);

	// Optional: init PCA9685 (not required for DC motor test)
	pwm.begin();
	pwm.setPWMFreq(60);

	// Ensure motors are stopped
	setMotorA(0);
	setMotorB(0);

	Serial.println(F("Motor test starting"));
}

void testMotorA() {
	Serial.println(F("A fwd"));
	setMotorA(200);
	delay(1500);
	Serial.println(F("A stop"));
	setMotorA(0);
	delay(500);
	Serial.println(F("A rev"));
	setMotorA(-200);
	delay(1500);
	Serial.println(F("A stop"));
	setMotorA(0);
	delay(1000);
}

void testMotorB() {
	Serial.println(F("B fwd"));
	setMotorB(200);
	delay(1500);
	Serial.println(F("B stop"));
	setMotorB(0);
	delay(500);
	Serial.println(F("B rev"));
	setMotorB(-200);
	delay(1500);
	Serial.println(F("B stop"));
	setMotorB(0);
	delay(1000);
}

void testBoth() {
	Serial.println(F("Both fwd"));
	setMotorA(200);
	setMotorB(200);
	delay(1500);
	Serial.println(F("Both stop"));
	setMotorA(0);
	setMotorB(0);
	delay(500);
	Serial.println(F("Both rev"));
	setMotorA(-200);
	setMotorB(-200);
	delay(1500);
	Serial.println(F("Both stop"));
	setMotorA(0);
	setMotorB(0);
	delay(1000);
}

void loop() {
	// Test the 7 servos on channels 0-6 in order
	sweepServo(0, F("0: neck rotate (left/right)"));
	sweepServo(1, F("1: neck top joint (up/down)"));
	sweepServo(2, F("2: neck bottom joint (up/down)"));
	sweepServo(3, F("3: right eye (up/down)"));
	sweepServo(4, F("4: left eye (up/down)"));
	sweepServo(5, F("5: right arm"));
	sweepServo(6, F("6: left arm"));

	// Then test motors
	testMotorA();
	testMotorB();
	testBoth();
}


