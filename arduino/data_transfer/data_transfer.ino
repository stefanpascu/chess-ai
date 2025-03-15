#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define MIN_PULSE 150  // Minim PWM pentru servo
#define MAX_PULSE 600  // Maxim PWM pentru servo
#define STEP 5         // Pas pentru mișcare graduală
#define SERVO_COUNT 6  // Numărul de servomotoare

// Pozițiile inițiale ale servomotoarelor (valorile PWM corespunzătoare poziției de repaus)
int servo_positions[SERVO_COUNT] = {375, 375, 375, 375, 375, 375};

void setup() {
  Serial.begin(9600);
  Wire.begin();               // Inițializează I2C
  pwm.begin();
  pwm.setPWMFreq(50);         // Setăm frecvența pentru servomotoare (50Hz)
  delay(10);                  // Mic delay pentru stabilizare
  Serial.println("Arduino ready");
}

void loop() {
  if (Serial.available()) {
    // Citim comanda până la newline
    String command = Serial.readStringUntil('\n');
    command.trim();  // Elimină spațiile suplimentare

    // Așteptăm o comandă în formatul: servo_index,target_angle
    // Ex: "2,120" va muta servo-ul 2 la 120°.
    int commaIndex = command.indexOf(',');
    if (commaIndex == -1) {
      Serial.println("Format comandă invalid");
      return;
    }

    int servo_index = command.substring(0, commaIndex).toInt();
    int targetAngle = command.substring(commaIndex + 1).toInt();

    if (servo_index < 0 || servo_index >= SERVO_COUNT) {
      Serial.println("Index servo invalid");
      return;
    }

    // Mapăm unghiul (0-180°) la valorile PWM (MIN_PULSE - MAX_PULSE)
    int targetPWM = map(targetAngle, 0, 180, MIN_PULSE, MAX_PULSE);

    // Mutăm servo-ul gradual către poziția țintă
    moveServo(servo_index, targetPWM);
  }
}

void moveServo(int servo, int targetPWM) {
  // Mută servo_positions[servo] treptat către targetPWM în pași de STEP
  while (servo_positions[servo] != targetPWM) {
    if (servo_positions[servo] < targetPWM) {
      servo_positions[servo] += STEP;
      if (servo_positions[servo] > targetPWM) {
        servo_positions[servo] = targetPWM;
      }
    } else {
      servo_positions[servo] -= STEP;
      if (servo_positions[servo] < targetPWM) {
        servo_positions[servo] = targetPWM;
      }
    }
    pwm.setPWM(servo, 0, servo_positions[servo]);
    delay(5);  // Delay pentru o tranziție lină
  }
  // După finalizarea mișcării, trimitem un mesaj de confirmare către Python
  Serial.println("MOVE_DONE");
}
