#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVO_MIN 150  // Минимальное значение (0°)
#define SERVO_MAX 600  // Максимальное значение (180°)

void setup() {
  Serial.begin(9600);
  pwm.begin();
  pwm.setPWMFreq(50);  // Частота для сервоприводов (50 Гц)
  delay(10);
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');

    // Ожидаем команду вида: SERVO:<канал>:<угол>
    if (input.startsWith("SERVO:")) {
      int first_colon = input.indexOf(':');
      int second_colon = input.indexOf(':', first_colon + 1);

      if (second_colon > 0) {
        int channel = input.substring(first_colon + 1, second_colon).toInt();
        int angle = input.substring(second_colon + 1).toInt();

        angle = constrain(angle, 0, 180);
        int pulse = map(angle, 0, 180, SERVO_MIN, SERVO_MAX);
        pwm.setPWM(channel, 0, pulse);
      }
    }
  }
}
