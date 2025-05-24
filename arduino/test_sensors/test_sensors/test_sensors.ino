const int SIG_PIN_1 = 6;  // Signal pin for the first mux
const int SIG_PIN_2 = 7;  // Signal pin for the second mux
const int SIG_PIN_3 = 8;  // Signal pin for the third mux
const int SIG_PIN_4 = 9;  // Signal pin for the fourth mux

const int SIG_PINS[4] = { 6, 7, 8, 9 };
const int ADDR_PINS[4][4] = {
  {22, 24, 26, 28},
  {23, 25, 27, 29},
  {30, 32, 34, 36},
  {31, 33, 35, 37}
};

void setup() {
  Serial.begin(9600);
  for (int mux = 0; mux < 4; mux++)
    for (int b = 0; b < 4; b++)
      pinMode(ADDR_PINS[mux][b], OUTPUT);

  for (int mux = 0; mux < 4; mux++)
    pinMode(SIG_PINS[mux], INPUT);
}

void loop() {
  bool first_print = true;
  int count = 0;
  for (int mux = 0; mux < 4; mux++) {
    // Inner loop over the 16 channels on that MUX
    for (int channel = 0; channel < 16; channel++) {
      // Drive the 4 address pins for this channel
      for (int b = 0; b < 4; b++) {
        digitalWrite(ADDR_PINS[mux][b], bitRead(channel, b));
      }

      // Read and immediately print
      int v = digitalRead(SIG_PINS[mux]);
      count++;
      if (first_print) {
        Serial.print(v);
        first_print = false;
      } else {
        Serial.print(' ');
        Serial.print(v);
      }
      if (count % 8 == 0) {
        Serial.println();
        count = 0;
      }
    }
  }
  Serial.println();
  delay(1500);
}

