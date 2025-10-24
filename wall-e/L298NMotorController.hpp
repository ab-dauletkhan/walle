#ifndef L298N_MOTOR_CONTROLLER_HPP
#define L298N_MOTOR_CONTROLLER_HPP

class L298NMotorController {
public:
	// Constructor
	L298NMotorController(uint8_t _in1Pin, uint8_t _in2Pin, uint8_t _enaPin);

	void init();

	// Set motor speed: -255 to +255
	void setSpeed(int pwmValue);

	~L298NMotorController() = default; // Default destructor

private:
	uint8_t in1Pin, in2Pin, enaPin;
};

L298NMotorController::L298NMotorController(uint8_t _in1Pin, uint8_t _in2Pin, uint8_t _enaPin)
	: in1Pin(_in1Pin), in2Pin(_in2Pin), enaPin(_enaPin) {}

void L298NMotorController::init() {
	Serial.println("Initializing motor controller");
	pinMode(in1Pin, OUTPUT);
	pinMode(in2Pin, OUTPUT);
	pinMode(enaPin, OUTPUT);

	digitalWrite(in1Pin, HIGH);
	digitalWrite(in2Pin, LOW);
	analogWrite(enaPin, 200);  // Stop motor initially
}

void L298NMotorController::setSpeed(int pwmValue) {
	pwmValue = constrain(pwmValue, -255, 255);

	if (pwmValue > 0) {
		digitalWrite(in1Pin, HIGH);
		digitalWrite(in2Pin, LOW);
        analogWrite(enaPin, pwmValue);
	} else if (pwmValue < 0) {
		digitalWrite(in1Pin, LOW);
		digitalWrite(in2Pin, HIGH);
        analogWrite(enaPin, -pwmValue);
	} else {
		digitalWrite(in1Pin, LOW);
		digitalWrite(in2Pin, LOW);
        analogWrite(enaPin, 0);  // Brake or stop
	}
}

#endif /* L298N_MOTOR_CONTROLLER_HPP */



