#include <Wire.h>

#define MPU_ADDR 0x68
#define SDA_PIN 21
#define SCL_PIN 22

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000); // Stable I2C speed

  // Wake up the MPU6500 (Clear sleep mode)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); 
  Wire.write(0x00); 
  if (Wire.endTransmission() == 0) {
    Serial.println("--- MPU6500 ONLINE ---");
  } else {
    Serial.println("--- SENSOR ERROR ---");
    while(1);
  }
}

void loop() {
  // Start reading from register 0x3B (Accel X High byte)
  // Requesting 14 bytes captures Accel (6B), Temp (2B), and Gyro (6B)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14);

  if (Wire.available() == 14) {
    // Read Accelerometer Raw Data
    int16_t ax = (Wire.read() << 8) | Wire.read();
    int16_t ay = (Wire.read() << 8) | Wire.read();
    int16_t az = (Wire.read() << 8) | Wire.read();
    
    // Skip Temperature Bytes (2 bytes)
    int16_t raw_temp = (Wire.read() << 8) | Wire.read();
    
    // Read Gyroscope Raw Data
    int16_t gx = (Wire.read() << 8) | Wire.read();
    int16_t gy = (Wire.read() << 8) | Wire.read();
    int16_t gz = (Wire.read() << 8) | Wire.read();

    // Math Conversions
    // Accelerometer: Divided by 16384.0 for +/- 2g range
    float accelX = ax / 16384.0; 
    float accelY = ay / 16384.0;
    float accelZ = az / 16384.0;

    // Gyroscope: Divided by 131.0 for +/- 250 deg/s range (Default)
    float gyroX = gx / 131.0;
    float gyroY = gy / 131.0;
    float gyroZ = gz / 131.0;

    // Print Accelerometer Data
    Serial.print("ACCEL [G] -> X: "); Serial.print(accelX, 2);Serial.print(",");
    Serial.print(" | Y: "); Serial.print(accelY, 2);Serial.print(",");
    Serial.print(" | Z: "); Serial.print(accelZ, 2);Serial.print(",");
    
    // Print Gyroscope Data
    Serial.print("  ||  GYRO [deg/s] -> X: "); Serial.print(gyroX, 1);Serial.print(",");
    Serial.print(" | Y: "); Serial.print(gyroY, 1);Serial.print(",");
    Serial.print(" | Z: "); Serial.println(gyroZ, 1);Serial.print(",");
  }

  delay(200); // 5 updates per second
}
