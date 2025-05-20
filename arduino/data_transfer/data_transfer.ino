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

int servo_pwm[SERVO_COUNT]  = {315, 310, 310, 386, 105, 211};
int target_pwm[SERVO_COUNT];
int speedDelay = 20;

const int SIG_PINS[4] = { 6, 7, 8, 9 };
const int ADDR_PINS[4][4] = {
  {22, 24, 26, 28},
  {23, 25, 27, 29},
  {30, 32, 34, 36},
  {31, 33, 35, 37}
};

// Maximum change in PWM value per update
#define MAX_PWM_CHANGE 3
#define SERVO_SPEED 1

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
  for (int i = 0; i < SERVO_COUNT; i++) {
    pwm.setPWM(i, 0, servo_pwm[i]);
  }
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
        Serial.println("ALL_SERVOS_DONE");
        break;
      } else if (cmd.startsWith("C")) {
        break;
      } else {
        parseAndSetTargets(cmd);
        smoothMoveSequential();
        Serial.println("MOVE_DONE");
      }
    }
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
}

void enterAiTurn() {
  currentTurn = AI_TURN;
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED,   HIGH);
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

void smoothMoveSequential() {
  int firstServo = (target_pwm[1] >= servo_pwm[1]) ? 1 : 2;
  
  // Move each servo one at a time until all servos reach their target positions
  for (int step = 0; step < SERVO_COUNT; step++) {
    int i = (firstServo + step) % SERVO_COUNT;
    
    // Calculate step direction and size
    int error = target_pwm[i] - servo_pwm[i];
    
    // Skip if already at target position
    if (error == 0) {
      continue;
    }
    
    // Determine direction and use a constant speed
    int direction = (error > 0) ? 1 : -1;
    int constantSpeed = SERVO_SPEED;
    
    // Move this servo until it reaches its target
    while (servo_pwm[i] != target_pwm[i]) {
      // Calculate the step size (using a constant speed)
      int step_size = min(constantSpeed, abs(target_pwm[i] - servo_pwm[i]));
      
      // Update servo position
      servo_pwm[i] += direction * step_size;
      
      // Apply the new position
      pwm.setPWM(i, 0, servo_pwm[i]);
      
      // Delay between steps
      delay(speedDelay);
    }
    
    // Extra delay between servo movements
    delay(100);
  }
}