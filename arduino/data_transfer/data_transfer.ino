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
Turn current_turn = PLAYER_TURN;

bool player_turn = true;

#define MIN_PULSE     150
#define MAX_PULSE     600
#define SERVO_COUNT   6

#define START_SERVO   1

int servo_pwm[SERVO_COUNT]  = {315, 310, 310, 386, 105, 211};
int target_pwm[SERVO_COUNT];
int speed_delay = 20;

const int SIG_PINS[4] = { 6, 7, 8, 9 };
const int ADDR_PINS[4][4] = {
  {22, 24, 26, 28},
  {23, 25, 27, 29},
  {30, 32, 34, 36},
  {31, 33, 35, 37}
};

#define MAX_PWM_CHANGE 3
#define SERVO_SPEED 1

struct ServoCalibration {
  int min_pulse;
  int max_pulse;
  int min_angle;
  int max_angle;
  bool reversed;
};

ServoCalibration servoSettings[SERVO_COUNT] = {
  {115, 515, 0, 180, false},
  {105, 515, 0, 180, false},
  {105, 515, 0, 180, false},
  {105, 515, 0, 180, false},
  {105, 480, 0, 180, false},
  {165, 578, 0, 180, false}
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
  pinMode(END_TURN_BUTTON, INPUT_PULLUP);

  digitalWrite(GREEN_LED, HIGH);
  digitalWrite(RED_LED,   LOW);

  delay(10);
  Serial.println("Arduino ready");

  enterPlayerTurn();
}

void loop() {
  if (current_turn == PLAYER_TURN) {
    while (digitalRead(END_TURN_BUTTON) == HIGH) { /* se asteapta apasarea butonului */ }
    delay(DEBOUNCE_MS);
    while (digitalRead(END_TURN_BUTTON) == LOW)  { /* se asteapta eliberarea butonului */ }
    delay(DEBOUNCE_MS);

    enterAiTurn();
  }
  else {
    ai_flow();
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
      } else if (cmd.startsWith("W")){
        player_win();
      } else if (cmd.startsWith("D")){
        player_defeat();
      } else if (cmd.startsWith("S")){
        player_stalemate();
      } else {
        parseAndSetTargets(cmd);
        smoothMoveSequential();
        Serial.println("MOVE_DONE");
      }
    }
}

void player_win() {
  for(int i = 0; i < 5; i++) {
    digitalWrite(GREEN_LED, HIGH);
    digitalWrite(RED_LED,   LOW);
    delay(1000);
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED,   LOW);
    delay(1000);
  }
  digitalWrite(GREEN_LED, HIGH);
}

void player_defeat() {
  for(int i = 0; i < 5; i++) {
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED,   HIGH);
    delay(1000);
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED,   LOW);
    delay(1000);
  }
  digitalWrite(RED_LED,   HIGH);
}

void player_stalemate() {
  for(int i = 0; i < 5; i++) {
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED,   HIGH);
    delay(1000);
    digitalWrite(GREEN_LED, HIGH);
    digitalWrite(RED_LED,   LOW);
    delay(1000);
  }
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED,   LOW);
}

String waitForResponse() {
  while (Serial.available() == 0) {
    /* se asteapta apasarea butonului */
  }
  String resp = Serial.readStringUntil('\n');
  resp.trim();
  return resp;
}

void enterPlayerTurn() {
  current_turn = PLAYER_TURN;
  digitalWrite(GREEN_LED, HIGH);
  digitalWrite(RED_LED,   LOW);
}

void enterAiTurn() {
  current_turn = AI_TURN;
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED,   HIGH);
}

void read_sensors() {
  bool first_print = true;

  for (int mux = 0; mux < 4; mux++) {
    for (int channel = 0; channel < 16; channel++) {
      for (int b = 0; b < 4; b++) {
        digitalWrite(ADDR_PINS[mux][b], bitRead(channel, b));
      }
      delay(5);

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
      target_pwm[idx++] = map(angle, 0, 180, servoSettings[idx].min_pulse, servoSettings[idx].max_pulse);
      last = i + 1;
    }
  }
}

void smoothMoveSequential() {
  int first_servo = (target_pwm[1] >= servo_pwm[1]) ? 1 : 2;
  
  for (int step = 0; step < SERVO_COUNT; step++) {
    int i = (first_servo + step) % SERVO_COUNT;
    
    int error = target_pwm[i] - servo_pwm[i];
    
    if (error == 0) {
      continue;
    }
    
    int direction = (error > 0) ? 1 : -1;
    int constant_speed = SERVO_SPEED;
    
    while (servo_pwm[i] != target_pwm[i]) {
      int step_size = min(constant_speed, abs(target_pwm[i] - servo_pwm[i]));
      servo_pwm[i] += direction * step_size;
      
      pwm.setPWM(i, 0, servo_pwm[i]);
      
      delay(speed_delay);
    }
    
    delay(100);
  }
}