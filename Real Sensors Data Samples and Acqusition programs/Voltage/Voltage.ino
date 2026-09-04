// Choose an ADC1 pin on the ESP32 (GPIO 34, 35, 36, or 39 are ideal input-only pins)
const int VOLTAGE_SENSOR_PIN = 34; 
// Internal resistor values of your 5:1 voltage sensor module
const float R1 = 30000.0; // 30k Ohms
const float R2 = 7500.0;  // 7.5k Ohms

// ESP32 uses 3.3V internal reference logic
const float V_REF = 3.3;

void setup() {
  Serial.begin(115200); // ESP32 standard high-speed baud rate
  delay(1000);          // Short pause to stabilize serial connection
  
  // Print the CSV Header Row (Time is in milliseconds since boot)
  Serial.println("Timestamp_ms,Raw_ADC,Measured_Voltage_V");
}

void loop() {
  // 1. ESP32 has a 12-bit ADC (Returns raw steps from 0 to 4095)
  int rawADC = analogRead(VOLTAGE_SENSOR_PIN);
  
  // 2. Convert the 12-bit digital value back to voltage (0 to 3.3V)
  float vOut = (rawADC * V_REF) / 4095.0;
  
  // 3. Reverse the 5:1 step-down ratio to find the battery voltage
  float inputVoltage = vOut / (R2 / (R1 + R2));
  
  // Get the current running timestamp
  unsigned long timestamp = millis();
  
  // 4. Print values separated strictly by commas
  Serial.print(timestamp);
  Serial.print(",");
  Serial.print(rawADC);
  Serial.print(",");
  Serial.println(inputVoltage, 2); // println ends the CSV row with a newline
  
  delay(500);
}
