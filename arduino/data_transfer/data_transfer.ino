#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define MIN_PULSE   150      // PWM minim (poziție 0°)
#define MAX_PULSE   600      // PWM maxim (poziție 180°)
#define SERVO_COUNT 6        // Număr de servomotoare

#define STEP_DELAY  1       // Delay între update-uri (ms)
#define TOTAL_STEPS 1500    // Numărul total de pași pentru deplasarea completă

// Pozițiile actuale ale servomotoarelor
int servo_positions[SERVO_COUNT] = {375, 375, 375, 375, 375, 375};
// Vom reține poziția de start pentru fiecare mișcare
int initial_pwm[SERVO_COUNT];
// Pozițiile țintă calculate (în termeni PWM)
int target_pwm[SERVO_COUNT];

unsigned long lastStepTime = 0;
int stepCount = 0;
bool moving = false;

void setup() {
  Serial.begin(9600);
  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);
  Serial.println("Arduino ready");
}

void loop() {
  // Dacă se primește o comandă de la Serial
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    parseAndSetTargets(cmd);
    stepCount = 0;
    moving = true;
    // Copiem pozițiile curente ca poziții de start pentru mișcare
    for (int i = 0; i < SERVO_COUNT; i++) {
      initial_pwm[i] = servo_positions[i];
    }
  }

  // Dacă suntem în mișcare și a trecut timpul de update
  if (moving && millis() - lastStepTime >= STEP_DELAY) {
    lastStepTime = millis();
    moving = updateServos();
    if (!moving) {
      Serial.println("MOVE_DONE");
    }
  }
}

// Funcția parse: așteaptă un șir de comenzi în formatul "angle0,angle1,...,angle5"
// Exemplu: "90,90,90,90,90,90"
void parseAndSetTargets(const String &command) {
  int idx = 0, last = 0;
  for (int i = 0; i <= command.length() && idx < SERVO_COUNT; i++) {
    if (i == command.length() || command.charAt(i) == ',') {
      int angle = command.substring(last, i).toInt();
      // Convertim unghiul la PWM, folosind map și constrain pentru a ne asigura că rămâne între 0 și 180
      target_pwm[idx++] = map(constrain(angle, 0, 180), 0, 180, MIN_PULSE, MAX_PULSE);
      last = i + 1;
    }
  }
}

// Funcția update: calculează poziția intermediară pentru fiecare servo folosind interpolare sinuoasă
bool updateServos() {
  // Calculăm progresul (0.0 - 1.0)
  float progress = (float)stepCount / TOTAL_STEPS;
  if (progress > 1.0) progress = 1.0;
  // Funcția de easing: începe de la 0 și ajunge la 1, cu o accelerație și decelerație lină
  float ease = 0.5 - 0.5 * cos(progress * PI);
  
  // Actualizăm fiecare servo
  for (int i = 0; i < SERVO_COUNT; i++) {
    int newPWM = initial_pwm[i] + (target_pwm[i] - initial_pwm[i]) * ease;
    servo_positions[i] = newPWM;
    pwm.setPWM(i, 0, newPWM);
  }
  
  stepCount++;
  // Când progresul ajunge la 1, mișcarea s-a terminat
  return progress < 1.0;
}
