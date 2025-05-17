#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

const int SIG_PIN_1 = 6;
const int SIG_PIN_2 = 7;
const int SIG_PIN_3 = 8;
const int SIG_PIN_4 = 9;

const int RED_LED         = 10;
const int GREEN_LED       = 11;
const int END_TURN_BUTTON = 12;
const int DEBOUNCE_MS     = 50;

enum Turn { PLAYER_TURN, AI_TURN };
Turn currentTurn = PLAYER_TURN;

int buttonState = 0;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;
bool player_turn = true;

#define MIN_PULSE     150
#define MAX_PULSE     600
#define SERVO_COUNT    6

// index of the servo you want to move first (0 = first element in your parsed list)
#define START_SERVO    1

int servo_pwm[SERVO_COUNT]  = {375, 375, 375, 375, 375, 375};
int target_pwm[SERVO_COUNT];
int speedDelay = 20;

const int SIG_PINS[4] = { 6, 7, 8, 9 };
const int ADDR_PINS[4][4] = {
  {22, 24, 26, 28},
  {23, 25, 27, 29},
  {30, 32, 34, 36},
  {31, 33, 35, 37}
};

// PID Controller parameters
float Kp[SERVO_COUNT] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};  // Proportional gain
float Ki[SERVO_COUNT] = {0.005, 0.005, 0.005, 0.005, 0.005, 0.005};  // Integral gain
float Kd[SERVO_COUNT] = {0.02, 0.02, 0.02, 0.02, 0.02, 0.02};  // Derivative gain

// PID variables for each servo
float previous_error[SERVO_COUNT] = {0, 0, 0, 0, 0, 0};
float integral[SERVO_COUNT] = {0, 0, 0, 0, 0, 0};
unsigned long last_time[SERVO_COUNT] = {0, 0, 0, 0, 0, 0};
float dt = 0.02;  // 20ms default

// Maximum change in PWM value per update
#define MAX_PWM_CHANGE 3

struct ServoCalibration {
  int minPulse;
  int maxPulse;
  int minAngle;
  int maxAngle;
  bool reversed;
};

ServoCalibration servoSettings[SERVO_COUNT] = {
  {115, 515, 0, 180, false},  // Servo 0
  {105, 515, 0, 180, false},  // Servo 1
  {105, 515, 0, 180, false},  // Servo 2
  {105, 515, 0, 180, false},  // Servo 3
  {105, 480, 0, 180, false},  // Servo 4
  {165, 578, 0, 180, false}   // Servo 5
};

void setup() {
  Serial.begin(9600);
  Wire.begin();
  pwm.begin();
  for (int mux = 0; mux < 4; mux++)
    for (int b = 0; b < 4; b++)
      pinMode(ADDR_PINS[mux][b], OUTPUT);

  for (int mux = 0; mux < 4; mux++)
    pinMode(SIG_PINS[mux], INPUT);
  pwm.setPWMFreq(50);
  player_turn = true;

  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED,   OUTPUT);
  pinMode(END_TURN_BUTTON, INPUT_PULLUP);  // button to GND

  // start with green on (ready), red off
  digitalWrite(GREEN_LED, HIGH);
  digitalWrite(RED_LED,   LOW);

  delay(10);
  Serial.println("Arduino ready");

  // Start in player-turn
  enterPlayerTurn();
}

void loop() {
  if (currentTurn == PLAYER_TURN) {
    // BLOCK here until the player taps the button
    while (digitalRead(END_TURN_BUTTON) == HIGH) { /* waiting for press */ }
    delay(DEBOUNCE_MS);
    while (digitalRead(END_TURN_BUTTON) == LOW)  { /* waiting for release */ }
    delay(DEBOUNCE_MS);

    // Now switch to the AI’s turn
    enterAiTurn();
  }
  else {  // AI_TURN
    ai_flow();

    // Immediately hand control back to the player
    enterPlayerTurn();
  }
}

void ai_flow() {
    read_sensors();
    while (true) {
      String cmd = waitForResponse();
      if (cmd.startsWith("A")) {
        break;
      } else if (cmd.startsWith("C")) {
        break;
      } else {
        // parseAndSetTargets(cmd);
        // smoothMoveSequentialPID();
        Serial.println("MOVE_DONE");
      }
    } 
    // delay(5000);              // ← simulate your AI/servo‐move
}

String waitForResponse() {
  // Wait until there’s at least one byte in the buffer
  while (Serial.available() == 0) {
    // nothing here — you could blink an LED or call yield() if you need background work
  }
  // Read until newline (you can change to '\r' or whatever your PC is sending)
  String resp = Serial.readStringUntil('\n');
  resp.trim();   // remove CR/LF or stray spaces
  return resp;
}

// Helpers to centralize LED + state logic
void enterPlayerTurn() {
  currentTurn = PLAYER_TURN;
  digitalWrite(GREEN_LED, HIGH);
  digitalWrite(RED_LED,   LOW);
  // Serial.println("HUMAN at play…");
}

void enterAiTurn() {
  currentTurn = AI_TURN;
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED,   HIGH);
  // Serial.println("AI at play…");  // done inside loop if you prefer
}

void read_sensors() {
  bool first_print = true;

  // Outer loop over each MUX
  for (int mux = 0; mux < 4; mux++) {
    // Inner loop over the 16 channels on that MUX
    for (int channel = 0; channel < 16; channel++) {
      // Drive the 4 address pins for this channel
      for (int b = 0; b < 4; b++) {
        digitalWrite(ADDR_PINS[mux][b], bitRead(channel, b));
      }
      delay(5);

      // Read and immediately print
      int v = digitalRead(SIG_PINS[mux]);
      if (first_print) {
        Serial.print(v);
        first_print = false;
      } else {
        Serial.print(' ');
        Serial.print(v);
      }
    }
  }
  Serial.println();
}

void parseAndSetTargets(const String &command) {
  int idx = 0, last = 0;
  for (int i = 0; i <= command.length() && idx < SERVO_COUNT; i++) {
    if (i == command.length() || command.charAt(i) == ',') {
      int angle = command.substring(last, i).toInt();
      angle = constrain(angle, 0, 180);
      target_pwm[idx++] = map(angle, 0, 180, servoSettings[idx].minPulse, servoSettings[idx].maxPulse);
      last = i + 1;
    }
  }
}

// Sequential mover with PID control
void smoothMoveSequentialPID() {
  bool all_servos_done = false;
  
  // Reset PID variables before movement starts
  for (int i = 0; i < SERVO_COUNT; i++) {
    integral[i] = 0;
    previous_error[i] = 0;
    last_time[i] = 0;
  }

  int firstServo = (target_pwm[1] >= servo_pwm[1]) ? 1 : 2;
  
  while (!all_servos_done) {
    all_servos_done = true;
    
    for (int step = 0; step < SERVO_COUNT; step++) {
      int i = (firstServo + step) % SERVO_COUNT;
      
      // Calculate error
      float error = target_pwm[i] - servo_pwm[i];
      
      // Only update if not at target position
      if (abs(error) > MAX_PWM_CHANGE) {
        all_servos_done = false;
        
        // Calculate PID output
        float pid_output = calculatePID(i, error);
        
        // Update servo position
        servo_pwm[i] += (int)pid_output;
        servo_pwm[i] = constrain(servo_pwm[i], servoSettings[i].minPulse, servoSettings[i].maxPulse);

        // Serial.print("S"); Serial.print(i);
        // Serial.print(" i=");   Serial.print(i);
        // Serial.print(" err=");   Serial.print(error);
        // Serial.print(" out=");   Serial.print(pid_output);
        // Serial.print(" pos=");   Serial.println(servo_pwm[i]);
      } else {
        // Serial.print("S"); Serial.print(i);
        // Serial.println(" at target");

        // close enough: snap to final target
        servo_pwm[i] = target_pwm[i];
      }
      // Apply the new position
      pwm.setPWM(i, 0, servo_pwm[i]);
      delay(10);
    }
    
    delay(speedDelay);
  }
}

void setPIDValues(const String &params) {
  int commaIndex1 = params.indexOf(',');
  int commaIndex2 = params.indexOf(',', commaIndex1 + 1);
  int commaIndex3 = params.indexOf(',', commaIndex2 + 1);
  
  if (commaIndex1 > 0 && commaIndex2 > 0 && commaIndex3 > 0) {
    int servoIdx = params.substring(0, commaIndex1).toInt();
    float p_gain = params.substring(commaIndex1 + 1, commaIndex2).toFloat();
    float i_gain = params.substring(commaIndex2 + 1, commaIndex3).toFloat();
    float d_gain = params.substring(commaIndex3 + 1).toFloat();
    
    if (servoIdx >= 0 && servoIdx < SERVO_COUNT) {
      Kp[servoIdx] = p_gain;
      Ki[servoIdx] = i_gain;
      Kd[servoIdx] = d_gain;
      
      // Reset PID variables when changing gains
      previous_error[servoIdx] = 0;
      integral[servoIdx] = 0;
      
      // Serial.print("PID values for servo ");
      // Serial.print(servoIdx);
      // Serial.print(" set to P=");
      // Serial.print(p_gain);
      // Serial.print(", I=");
      // Serial.print(i_gain);
      // Serial.print(", D=");
      // Serial.println(d_gain);
    }
  }
}

// Calculate PID output for a servo
float calculatePID(int servoIdx, float error) {
  unsigned long current_time = millis();
  float dt_actual = (current_time - last_time[servoIdx]) / 1000.0;
  
  // Use the default dt if it's the first calculation or if time difference is too small
  if (last_time[servoIdx] == 0 || dt_actual < 0.001) {
    dt_actual = dt;
  }
  
  // Calculate integral with anti-windup
  integral[servoIdx] += error * dt_actual;
  integral[servoIdx] = constrain(integral[servoIdx], -100, 100);  // Prevent integral windup
  
  // Calculate derivative
  float derivative = (error - previous_error[servoIdx]) / dt_actual;
  
  // PID formula
  float output = Kp[servoIdx] * error + 
                 Ki[servoIdx] * integral[servoIdx] + 
                 Kd[servoIdx] * derivative;
  
  // Store current values for next iteration
  previous_error[servoIdx] = error;
  last_time[servoIdx] = current_time;
  
  // Limit the maximum change in PWM value
  return constrain(output, -MAX_PWM_CHANGE, MAX_PWM_CHANGE);
}