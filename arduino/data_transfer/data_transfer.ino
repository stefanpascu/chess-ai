#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define MIN_PULSE 150  // Minim PWM pentru servo
#define MAX_PULSE 600  // Maxim PWM pentru servo
#define STEP 1         // Pas pentru mișcare graduală
#define SERVO_COUNT 6  // Numărul de servomotoare

// Pozițiile inițiale ale servomotoarelor (valorile PWM corespunzătoare poziției de repaus)
int servo_positions[SERVO_COUNT] = {375, 375, 375, 375, 375, 375};

void setup() {
  Serial.begin(9600);
  Wire.begin();               // Inițializează I2C
  pwm.begin();
  pwm.setPWMFreq(50);         // Setăm frecvența pentru servomotoare (50Hz)
  delay(10);                  // Delay pentru stabilizare
  Serial.println("Arduino ready");
}

void loop() {
  if (Serial.available()) {
    // Citim comanda până la newline
    String command = Serial.readStringUntil('\n');
    command.trim();  // Elimină spațiile suplimentare

    // Așteptăm o comandă în formatul: angle0,angle1,...,angle5
    // Exemplu: "90,120,45,60,90,90"
    int commaCount = 0;
    for (unsigned int i = 0; i < command.length(); i++) {
      if (command.charAt(i) == ',') commaCount++;
    }
    
    if (commaCount != SERVO_COUNT - 1) {
      Serial.println("Format comandă invalid");
      return;
    }
    
    int angles[SERVO_COUNT];
    int lastIndex = 0;
    int servoIndex = 0;
    for (unsigned int i = 0; i < command.length(); i++) {
      if (command.charAt(i) == ',' || i == command.length() - 1) {
        String value;
        if (command.charAt(i) == ',') {
          value = command.substring(lastIndex, i);
          lastIndex = i + 1;
        } else {
          value = command.substring(lastIndex, i + 1);
        }
        angles[servoIndex] = value.toInt();
        servoIndex++;
        if (servoIndex >= SERVO_COUNT) break;
      }
    }
    
    // Mișcăm simultan toate servomotoarele către unghiurile țintă
    bool moving = true;
    while(moving) {
      moving = false;
      for (int i = 0; i < SERVO_COUNT; i++) {
        int targetPWM = map(angles[i], 0, 180, MIN_PULSE, MAX_PULSE);
        if (servo_positions[i] < targetPWM) {
          servo_positions[i] += STEP;
          if (servo_positions[i] > targetPWM) servo_positions[i] = targetPWM;
          moving = true;
        } else if (servo_positions[i] > targetPWM) {
          servo_positions[i] -= STEP;
          if (servo_positions[i] < targetPWM) servo_positions[i] = targetPWM;
          moving = true;
        }
        pwm.setPWM(i, 0, servo_positions[i]);
      }
      delay(2);
    }
    Serial.println("MOVE_DONE");
  }
}
