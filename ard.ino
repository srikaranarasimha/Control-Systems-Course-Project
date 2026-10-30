const byte IN1 = 7;
const byte IN2 = 8;
const byte ENA = 9;
const byte POT = A0;

// These are calibrated analogRead() counts, not angles or physical units.
const int LEFT_POSITION = 911;
const int HOME_POSITION = 940;
const int RIGHT_POSITION = 964;

// PD tuning: Kp is PWM per potentiometer count; Kd is PWM per
// potentiometer count/second. PWM limits make the approach slowdown visible.
const float KP = 4.0f;
const float KD = 0.8f;
const float VELOCITY_FILTER_ALPHA = 0.25f;
const float MAX_MEASURED_VELOCITY = 250.0f;
const int MIN_PWM = 45;
const int MAX_PWM = 85;
const int DECELERATION_DISTANCE = 20;
const int POSITION_TOLERANCE = 1;
const unsigned long TARGET_CONFIRMATION_MS = 150;
const unsigned long CONTROL_INTERVAL_MS = 50;
const unsigned long MOVE_TIMEOUT_MS = 10000;
const unsigned long STATUS_INTERVAL_MS = 500;

bool moving = false;
bool settlingAtTarget = false;
int targetPosition = HOME_POSITION;
unsigned long moveStartedAt = 0;
unsigned long targetCandidateSince = 0;
unsigned long lastControlAt = 0;
unsigned long lastStatusAt = 0;
int appliedPwm = 0;
int previousPosition = HOME_POSITION;
float filteredVelocity = 0.0f;
float pTerm = 0.0f;
float dTerm = 0.0f;
float controlEffort = 0.0f;
float saturatedEffort = 0.0f;
int pwmLimit = 0;
const char* driveDirection = "off";

void stopMotor() {
  // Disable drive and clear the active move so another command can be accepted.
  analogWrite(ENA, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  moving = false;
  settlingAtTarget = false;
  appliedPwm = 0;
  pTerm = 0.0f;
  dTerm = 0.0f;
  controlEffort = 0.0f;
  saturatedEffort = 0.0f;
  pwmLimit = 0;
  driveDirection = "off";
}

void startMove(int target, unsigned long now) {
  stopMotor();
  targetPosition = target;
  moveStartedAt = now;
  settlingAtTarget = false;
  moving = true;
  lastControlAt = now;
  previousPosition = analogRead(POT);
  filteredVelocity = 0.0f;
  Serial.print(F("Moving to "));
  Serial.print(targetPosition);
  Serial.print(F(" kp="));
  Serial.print(KP, 1);
  Serial.print(F(" kd="));
  Serial.print(KD, 1);
  Serial.print(F(" min_pwm="));
  Serial.print(MIN_PWM);
  Serial.print(F(" max_pwm="));
  Serial.print(MAX_PWM);
  Serial.print(F(" decel_counts="));
  Serial.println(DECELERATION_DISTANCE);
}

void startHomeReset(unsigned long now) { startMove(HOME_POSITION, now); }

void readCommands(unsigned long now) {
  while (Serial.available() > 0) {
    const char command = Serial.read();
    if (command == '\r' || command == '\n') {
      continue;
    }

    if (command == 'S' || command == 's') {
      stopMotor();
      Serial.println(F("Stopped"));
    } else if (command == 'L' || command == 'l') {
      if (moving) {
        Serial.println(F("Replacing active move with LEFT target."));
      }
      startMove(LEFT_POSITION, now);
    } else if (command == 'H' || command == 'h') {
      if (moving) {
        Serial.println(F("Replacing active move with HOME reset."));
      }
      startHomeReset(now);
    } else if (command == 'R' || command == 'r') {
      if (moving) {
        Serial.println(F("Replacing active move with RIGHT target."));
      }
      startMove(RIGHT_POSITION, now);
    } else {
      Serial.println(F("Unknown command; use L, H, R, or S"));
    }
  }
}

void applyPDController(int target, int position, unsigned long now) {
  const int error = target - position;
  if (abs(error) <= POSITION_TOLERANCE) {
    analogWrite(ENA, 0);
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    appliedPwm = 0;
    pTerm = KP * error;
    dTerm = -KD * filteredVelocity;
    controlEffort = pTerm + dTerm;
    saturatedEffort = 0.0f;
    pwmLimit = 0;
    driveDirection = "off";
    return;
  }

  const unsigned long elapsedMs = now - lastControlAt;
  float measuredVelocity = (position - previousPosition) * 1000.0f / elapsedMs;
  measuredVelocity = constrain(measuredVelocity, -MAX_MEASURED_VELOCITY,
                               MAX_MEASURED_VELOCITY);
  filteredVelocity +=
      VELOCITY_FILTER_ALPHA * (measuredVelocity - filteredVelocity);
  previousPosition = position;

  pTerm = KP * error;
  dTerm = -KD * filteredVelocity;
  controlEffort = pTerm + dTerm;
  const int direction = error > 0 ? 1 : -1;
  const float effortTowardTarget = controlEffort * direction;
  const int distance = abs(error);
  const int profiledDistance =
      distance < DECELERATION_DISTANCE ? distance : DECELERATION_DISTANCE;
  pwmLimit =
      MIN_PWM + (MAX_PWM - MIN_PWM) * profiledDistance / DECELERATION_DISTANCE;
  saturatedEffort = constrain(effortTowardTarget, MIN_PWM, pwmLimit);
  if (direction > 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    driveDirection = "increasing";
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    driveDirection = "decreasing";
  }

  const int magnitude = static_cast<int>(saturatedEffort);
  analogWrite(ENA, magnitude);
  appliedPwm = magnitude;
}

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);
  stopMotor();

  Serial.begin(9600);
  Serial.println(F("PD_CONTROLLER_VERSION=4"));
  Serial.println(F("Baud=9600; commands: L=left, H=home, R=right, S=stop"));
  Serial.println(
      F("APPROACH_PROFILE: max PWM holds until 20 counts, then ramps to "
        "minimum"));
  Serial.println(
      F("Telemetry: t_ms position target error velocity v_toward P D u "
        "u_toward pwm_limit pwm direction mode"));
  startHomeReset(millis());
}

void loop() {
  const unsigned long now = millis();
  readCommands(now);

  const int position = analogRead(POT);
  if (moving) {
    // Timeout bounds movement if the target cannot be reached.
    if (now - moveStartedAt >= MOVE_TIMEOUT_MS) {
      stopMotor();
      Serial.println(F("ERROR: move timed out; motor stopped"));
    } else if (abs(targetPosition - position) <= POSITION_TOLERANCE) {
      if (!settlingAtTarget) {
        analogWrite(ENA, 0);
        digitalWrite(IN1, LOW);
        digitalWrite(IN2, LOW);
        appliedPwm = 0;
        driveDirection = "off";
        settlingAtTarget = true;
        targetCandidateSince = now;
      } else if (now - targetCandidateSince >= TARGET_CONFIRMATION_MS) {
        stopMotor();
        Serial.print(F("Target confirmed; final position="));
        Serial.println(position);
      }
    } else {
      if (settlingAtTarget) {
        settlingAtTarget = false;
        previousPosition = position;
        filteredVelocity = 0.0f;
        lastControlAt = now;
      }
      if (now - lastControlAt >= CONTROL_INTERVAL_MS) {
        applyPDController(targetPosition, position, now);
        lastControlAt = now;
      }
    }
  }

  if (now - lastStatusAt >= STATUS_INTERVAL_MS) {
    Serial.print(F("t_ms="));
    Serial.print(now);
    Serial.print(F(" position="));
    Serial.print(position);
    Serial.print(F(" target="));
    Serial.print(targetPosition);
    Serial.print(F(" error="));
    Serial.print(targetPosition - position);
    Serial.print(F(" velocity="));
    Serial.print(filteredVelocity);
    Serial.print(F(" v_toward="));
    Serial.print(filteredVelocity * (targetPosition >= position ? 1 : -1));
    Serial.print(F(" p="));
    Serial.print(pTerm);
    Serial.print(F(" d="));
    Serial.print(dTerm);
    Serial.print(F(" u="));
    Serial.print(controlEffort);
    Serial.print(F(" u_toward="));
    Serial.print(saturatedEffort);
    Serial.print(F(" pwm_limit="));
    Serial.print(pwmLimit);
    Serial.print(F(" pwm="));
    Serial.print(appliedPwm);
    Serial.print(F(" direction="));
    Serial.print(driveDirection);
    Serial.print(F(" mode="));
    Serial.print(settlingAtTarget ? F("settle")
                 : moving         ? F("move")
                                  : F("idle"));
    Serial.println();
    lastStatusAt = now;
  }
}
