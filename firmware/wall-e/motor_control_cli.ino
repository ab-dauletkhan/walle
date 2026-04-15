#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

// =========================
// Hardware mapping
// =========================
const uint8_t OE = 11;  // PCA9685 Output Enable, LOW = enabled
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

// Keep the motor_test sketch on the exact wiring that already worked on the robot.
// On some boards pins 2 and 7 are not hardware PWM, but we preserve this mapping
// because it matches the existing wiring and direction behavior in the old sketch.
const uint8_t LEFT_ENA = 2;
const uint8_t LEFT_IN1 = 3;
const uint8_t LEFT_IN2 = 4;
const uint8_t RIGHT_IN1 = 5;
const uint8_t RIGHT_IN2 = 6;
const uint8_t RIGHT_ENA = 7;

// =========================
// Safety / tuning
// =========================
const int MAX_MOTOR_SPEED = 200;
const unsigned long MOTOR_TIMEOUT_MS = 3000;
const unsigned long MAX_TIMED_MOVE_MS = 5000;
const uint8_t SERIAL_BUFFER_SIZE = 96;

// =========================
// Head pan servo only
// =========================
struct ServoAxis {
  uint8_t channel;
  int minTick;
  int maxTick;
  int centerTick;
  int currentTick;
  int maxDeltaPerCommand;
  const char* name;
};

// Keep the old motor_test calibration:
//   channel 2 = head horizontal turn
ServoAxis headPan = {2, 150, 550, 350, 350, 60, "head pan"};

// =========================
// Runtime state
// =========================
int leftMotorValue = 0;
int rightMotorValue = 0;
int motorTrim = 0;
unsigned long lastMotorCommandMs = 0;
unsigned long scheduledMotorStopMs = 0;

char serialBuffer[SERIAL_BUFFER_SIZE];
uint8_t serialLength = 0;

bool parseLongToken(const char* token, long& outValue) {
  if (token == NULL || *token == '\0') {
    return false;
  }

  char* endPtr = NULL;
  long parsed = strtol(token, &endPtr, 10);
  if (endPtr == token || *endPtr != '\0') {
    return false;
  }

  outValue = parsed;
  return true;
}

void lowercaseInPlace(char* text) {
  while (*text != '\0') {
    *text = char(tolower(*text));
    text++;
  }
}

int clampMotorSpeed(long speed) {
  return constrain((int)speed, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED);
}

bool applyServoTick(ServoAxis& axis, int targetTick) {
  if (targetTick < axis.minTick || targetTick > axis.maxTick) {
    Serial.print(F("Rejected: "));
    Serial.print(axis.name);
    Serial.print(F(" tick must be in "));
    Serial.print(axis.minTick);
    Serial.print(F(".."));
    Serial.println(axis.maxTick);
    return false;
  }

  if (abs(targetTick - axis.currentTick) > axis.maxDeltaPerCommand) {
    Serial.print(F("Rejected: "));
    Serial.print(axis.name);
    Serial.print(F(" jump too large. Max delta is "));
    Serial.println(axis.maxDeltaPerCommand);
    return false;
  }

  axis.currentTick = targetTick;
  pwm.setPWM(axis.channel, 0, targetTick);

  Serial.print(F("OK -> "));
  Serial.print(axis.name);
  Serial.print(F(" tick "));
  Serial.println(axis.currentTick);
  return true;
}

int percentToTick(const ServoAxis& axis, int percent) {
  percent = constrain(percent, 0, 100);
  return map(percent, 0, 100, axis.minTick, axis.maxTick);
}

int tickToPercent(const ServoAxis& axis, int tick) {
  return map(tick, axis.minTick, axis.maxTick, 0, 100);
}

void centerHead() {
  headPan.currentTick = headPan.centerTick;
  pwm.setPWM(headPan.channel, 0, headPan.centerTick);
}

void disableHeadServo() {
  pwm.setPWM(headPan.channel, 0, 0);
}

void applySingleMotor(uint8_t in1, uint8_t in2, uint8_t ena, int pwmValue) {
  pwmValue = clampMotorSpeed(pwmValue);

  if (pwmValue > 0) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    analogWrite(ena, pwmValue);
  } else if (pwmValue < 0) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    analogWrite(ena, -pwmValue);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    analogWrite(ena, 0);
  }
}

void stopMotors() {
  applySingleMotor(LEFT_IN1, LEFT_IN2, LEFT_ENA, 0);
  applySingleMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_ENA, 0);
  leftMotorValue = 0;
  rightMotorValue = 0;
  scheduledMotorStopMs = 0;
}

void applyDriveCommand(int leftSpeed, int rightSpeed, unsigned long durationMs) {
  int adjustedLeft = 0;
  int adjustedRight = 0;

  if (leftSpeed != 0) {
    adjustedLeft = clampMotorSpeed(leftSpeed - motorTrim);
  }

  if (rightSpeed != 0) {
    adjustedRight = clampMotorSpeed(rightSpeed + motorTrim);
  }

  applySingleMotor(LEFT_IN1, LEFT_IN2, LEFT_ENA, adjustedLeft);
  applySingleMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_ENA, adjustedRight);

  leftMotorValue = adjustedLeft;
  rightMotorValue = adjustedRight;
  lastMotorCommandMs = millis();

  if (durationMs > 0) {
    scheduledMotorStopMs = millis() + durationMs;
  } else {
    scheduledMotorStopMs = 0;
  }

  Serial.print(F("OK -> left="));
  Serial.print(leftMotorValue);
  Serial.print(F(" right="));
  Serial.print(rightMotorValue);

  if (durationMs > 0) {
    Serial.print(F(" for "));
    Serial.print(durationMs);
    Serial.print(F(" ms"));
  }

  Serial.println();
}

void printStatus() {
  Serial.println();
  Serial.println(F("Head:"));
  Serial.print(F("  channel = "));
  Serial.println(headPan.channel);
  Serial.print(F("  tick = "));
  Serial.println(headPan.currentTick);
  Serial.print(F("  percent = "));
  Serial.println(tickToPercent(headPan, headPan.currentTick));
  Serial.print(F("  center tick = "));
  Serial.println(headPan.centerTick);
  Serial.print(F("  range = "));
  Serial.print(headPan.minTick);
  Serial.print(F(".."));
  Serial.println(headPan.maxTick);

  Serial.println(F("Motors:"));
  Serial.print(F("  left = "));
  Serial.println(leftMotorValue);
  Serial.print(F("  right = "));
  Serial.println(rightMotorValue);
  Serial.print(F("  trim = "));
  Serial.println(motorTrim);
  Serial.print(F("  speed limit = "));
  Serial.println(MAX_MOTOR_SPEED);
  Serial.print(F("  watchdog timeout ms = "));
  Serial.println(MOTOR_TIMEOUT_MS);
  Serial.print(F("  max timed move ms = "));
  Serial.println(MAX_TIMED_MOVE_MS);
  Serial.println();
}

void printHelp() {
  Serial.println(F("Commands:"));
  Serial.println(F("  help                         -> show this help"));
  Serial.println(F("  status                       -> show head + motor state"));
  Serial.println(F("  home                         -> center the head"));
  Serial.println(F("  off                          -> disable head servo PWM"));
  Serial.println(F("  stop                         -> stop wheel motors"));
  Serial.println(F("  stopm                        -> legacy alias for stop"));
  Serial.println(F("  stopall                      -> stop motors and turn head servo off"));
  Serial.println(F("  trim VALUE                   -> set wheel trim (-50..50)"));
  Serial.println(F("  head center                  -> center the head"));
  Serial.println(F("  head pos PCT                 -> head pan by percent (0..100)"));
  Serial.println(F("  head tick TICK               -> head pan by raw PWM tick"));
  Serial.println(F("  head step DELTA              -> move head by relative tick delta"));
  Serial.println(F("  left SPEED [MS]              -> left wheel speed only"));
  Serial.println(F("  right SPEED [MS]             -> right wheel speed only"));
  Serial.println(F("  ma SPEED                     -> legacy alias for left SPEED"));
  Serial.println(F("  mb SPEED                     -> legacy alias for right SPEED"));
  Serial.println(F("  drive SPEED [MS]             -> both wheels same speed"));
  Serial.println(F("  both SPEED                   -> legacy alias for drive SPEED"));
  Serial.println(F("  spin SPEED [MS]              -> body rotate in place"));
  Serial.println(F("  tank LEFT RIGHT [MS]         -> direct wheel control"));
  Serial.println(F("  mix MOVE TURN [MS]           -> differential mix"));
  Serial.println();
  Serial.println(F("Notes:"));
  Serial.println(F("  SPEED range is -200..200."));
  Serial.println(F("  spin uses left=-speed and right=+speed."));
  Serial.println(F("  mix uses left=(move-turn), right=(move+turn)."));
  Serial.println(F("  MS is optional. Timed moves auto-stop when the timer ends."));
  Serial.println();
  Serial.println(F("Examples:"));
  Serial.println(F("  head center"));
  Serial.println(F("  head pos 20"));
  Serial.println(F("  head step -30"));
  Serial.println(F("  drive 120 800"));
  Serial.println(F("  spin 100 600"));
  Serial.println(F("  tank -80 80 500"));
  Serial.println(F("  mix 150 -40"));
  Serial.println();
}

bool parseOptionalDuration(char* token, unsigned long& durationMs) {
  durationMs = 0;
  if (token == NULL) {
    return true;
  }

  long rawDuration = 0;
  if (!parseLongToken(token, rawDuration) || rawDuration < 0) {
    Serial.println(F("Rejected: duration must be a positive integer in milliseconds."));
    return false;
  }

  if ((unsigned long)rawDuration > MAX_TIMED_MOVE_MS) {
    Serial.print(F("Rejected: duration exceeds max timed move of "));
    Serial.print(MAX_TIMED_MOVE_MS);
    Serial.println(F(" ms."));
    return false;
  }

  durationMs = (unsigned long)rawDuration;
  return true;
}

bool parseRequiredSpeed(char* token, int& speedOut) {
  long rawSpeed = 0;
  if (!parseLongToken(token, rawSpeed)) {
    Serial.println(F("Rejected: speed must be an integer."));
    return false;
  }

  if (rawSpeed < -MAX_MOTOR_SPEED || rawSpeed > MAX_MOTOR_SPEED) {
    Serial.print(F("Rejected: speed must be in "));
    Serial.print(-MAX_MOTOR_SPEED);
    Serial.print(F(".."));
    Serial.println(MAX_MOTOR_SPEED);
    return false;
  }

  speedOut = (int)rawSpeed;
  return true;
}

void handleHeadCommand(char* args) {
  char* savePtr = NULL;
  char* subcommand = strtok_r(args, " \t", &savePtr);

  if (subcommand == NULL) {
    Serial.println(F("Usage: head center | pos PCT | tick TICK | step DELTA"));
    return;
  }

  if (strcmp(subcommand, "center") == 0) {
    centerHead();
    Serial.println(F("OK -> head centered."));
    return;
  }

  if (strcmp(subcommand, "off") == 0) {
    disableHeadServo();
    Serial.println(F("OK -> head servo PWM off."));
    return;
  }

  char* valueToken = strtok_r(NULL, " \t", &savePtr);
  long rawValue = 0;

  if (!parseLongToken(valueToken, rawValue)) {
    Serial.println(F("Rejected: missing or invalid head value."));
    return;
  }

  if (strcmp(subcommand, "pos") == 0) {
    if (rawValue < 0 || rawValue > 100) {
      Serial.println(F("Rejected: head pos must be in 0..100."));
      return;
    }
    applyServoTick(headPan, percentToTick(headPan, (int)rawValue));
    return;
  }

  if (strcmp(subcommand, "tick") == 0) {
    applyServoTick(headPan, (int)rawValue);
    return;
  }

  if (strcmp(subcommand, "step") == 0) {
    if (abs((int)rawValue) > headPan.maxDeltaPerCommand) {
      Serial.print(F("Rejected: head step exceeds max delta of "));
      Serial.println(headPan.maxDeltaPerCommand);
      return;
    }
    applyServoTick(headPan, headPan.currentTick + (int)rawValue);
    return;
  }

  Serial.println(F("Unknown head command."));
}

void handleTrimCommand(char* args) {
  long rawTrim = 0;
  if (!parseLongToken(args, rawTrim)) {
    Serial.println(F("Usage: trim VALUE"));
    return;
  }

  if (rawTrim < -50 || rawTrim > 50) {
    Serial.println(F("Rejected: trim must be in -50..50."));
    return;
  }

  motorTrim = (int)rawTrim;
  Serial.print(F("OK -> motor trim set to "));
  Serial.println(motorTrim);
}

void handleSingleWheelCommand(char* args, bool leftWheel) {
  char* savePtr = NULL;
  char* speedToken = strtok_r(args, " \t", &savePtr);
  char* durationToken = strtok_r(NULL, " \t", &savePtr);
  int speed = 0;
  unsigned long durationMs = 0;

  if (!parseRequiredSpeed(speedToken, speed) ||
      !parseOptionalDuration(durationToken, durationMs)) {
    return;
  }

  if (leftWheel) {
    applyDriveCommand(speed, 0, durationMs);
  } else {
    applyDriveCommand(0, speed, durationMs);
  }
}

void handleDriveCommand(char* args) {
  char* savePtr = NULL;
  char* speedToken = strtok_r(args, " \t", &savePtr);
  char* durationToken = strtok_r(NULL, " \t", &savePtr);
  int speed = 0;
  unsigned long durationMs = 0;

  if (!parseRequiredSpeed(speedToken, speed) ||
      !parseOptionalDuration(durationToken, durationMs)) {
    return;
  }

  applyDriveCommand(speed, speed, durationMs);
}

void handleSpinCommand(char* args) {
  char* savePtr = NULL;
  char* speedToken = strtok_r(args, " \t", &savePtr);
  char* durationToken = strtok_r(NULL, " \t", &savePtr);
  int speed = 0;
  unsigned long durationMs = 0;

  if (!parseRequiredSpeed(speedToken, speed) ||
      !parseOptionalDuration(durationToken, durationMs)) {
    return;
  }

  applyDriveCommand(-speed, speed, durationMs);
}

void handleTankCommand(char* args) {
  char* savePtr = NULL;
  char* leftToken = strtok_r(args, " \t", &savePtr);
  char* rightToken = strtok_r(NULL, " \t", &savePtr);
  char* durationToken = strtok_r(NULL, " \t", &savePtr);
  int leftSpeed = 0;
  int rightSpeed = 0;
  unsigned long durationMs = 0;

  if (!parseRequiredSpeed(leftToken, leftSpeed) ||
      !parseRequiredSpeed(rightToken, rightSpeed) ||
      !parseOptionalDuration(durationToken, durationMs)) {
    return;
  }

  applyDriveCommand(leftSpeed, rightSpeed, durationMs);
}

void handleMixCommand(char* args) {
  char* savePtr = NULL;
  char* moveToken = strtok_r(args, " \t", &savePtr);
  char* turnToken = strtok_r(NULL, " \t", &savePtr);
  char* durationToken = strtok_r(NULL, " \t", &savePtr);
  int move = 0;
  int turn = 0;
  unsigned long durationMs = 0;

  if (!parseRequiredSpeed(moveToken, move) ||
      !parseRequiredSpeed(turnToken, turn) ||
      !parseOptionalDuration(durationToken, durationMs)) {
    return;
  }

  applyDriveCommand(move - turn, move + turn, durationMs);
}

void processCommand(char* line) {
  lowercaseInPlace(line);

  char* savePtr = NULL;
  char* command = strtok_r(line, " \t", &savePtr);
  char* args = savePtr;

  if (command == NULL) {
    return;
  }

  if (strcmp(command, "help") == 0) {
    printHelp();
  } else if (strcmp(command, "status") == 0) {
    printStatus();
  } else if (strcmp(command, "home") == 0) {
    centerHead();
    Serial.println(F("OK -> head centered."));
  } else if (strcmp(command, "off") == 0) {
    disableHeadServo();
    Serial.println(F("OK -> head servo PWM off."));
  } else if (strcmp(command, "stop") == 0) {
    stopMotors();
    Serial.println(F("OK -> motors stopped."));
  } else if (strcmp(command, "stopm") == 0) {
    stopMotors();
    Serial.println(F("OK -> motors stopped."));
  } else if (strcmp(command, "stopall") == 0) {
    stopMotors();
    disableHeadServo();
    Serial.println(F("OK -> motors stopped and head servo PWM off."));
  } else if (strcmp(command, "trim") == 0) {
    handleTrimCommand(args);
  } else if (strcmp(command, "head") == 0) {
    handleHeadCommand(args);
  } else if (strcmp(command, "left") == 0) {
    handleSingleWheelCommand(args, true);
  } else if (strcmp(command, "right") == 0) {
    handleSingleWheelCommand(args, false);
  } else if (strcmp(command, "ma") == 0) {
    handleSingleWheelCommand(args, true);
  } else if (strcmp(command, "mb") == 0) {
    handleSingleWheelCommand(args, false);
  } else if (strcmp(command, "drive") == 0) {
    handleDriveCommand(args);
  } else if (strcmp(command, "both") == 0) {
    handleDriveCommand(args);
  } else if (strcmp(command, "spin") == 0) {
    handleSpinCommand(args);
  } else if (strcmp(command, "tank") == 0) {
    handleTankCommand(args);
  } else if (strcmp(command, "mix") == 0) {
    handleMixCommand(args);
  } else {
    Serial.println(F("Unknown command."));
    printHelp();
  }
}

void pollSerialCommands() {
  while (Serial.available() > 0) {
    char incoming = (char)Serial.read();

    if (incoming == '\r') {
      continue;
    }

    if (incoming == '\n') {
      serialBuffer[serialLength] = '\0';
      processCommand(serialBuffer);
      serialLength = 0;
      continue;
    }

    if (serialLength < SERIAL_BUFFER_SIZE - 1) {
      serialBuffer[serialLength++] = incoming;
    } else {
      serialLength = 0;
      Serial.println(F("Rejected: command too long."));
    }
  }
}

void handleMotorTimeouts() {
  unsigned long now = millis();

  if (scheduledMotorStopMs > 0) {
    if (now >= scheduledMotorStopMs) {
      stopMotors();
      Serial.println(F("Timed move complete -> stopped."));
    }
    return;
  }

  if ((leftMotorValue != 0 || rightMotorValue != 0) &&
      (now - lastMotorCommandMs > MOTOR_TIMEOUT_MS)) {
    stopMotors();
    Serial.println(F("Motor watchdog timeout -> stopped."));
  }
}

// =========================
// Setup / loop
// =========================
void setup() {
  Serial.begin(115200);
  Wire.begin();

  pinMode(LEFT_IN1, OUTPUT);
  pinMode(LEFT_IN2, OUTPUT);
  pinMode(LEFT_ENA, OUTPUT);
  pinMode(RIGHT_IN1, OUTPUT);
  pinMode(RIGHT_IN2, OUTPUT);
  pinMode(RIGHT_ENA, OUTPUT);
  pinMode(OE, OUTPUT);

  digitalWrite(OE, LOW);

  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);

  stopMotors();
  centerHead();
  lastMotorCommandMs = millis();

  Serial.println(F("WALL-E head + wheel test controller started."));
  printHelp();
  printStatus();
}

void loop() {
  handleMotorTimeouts();
  pollSerialCommands();
}
