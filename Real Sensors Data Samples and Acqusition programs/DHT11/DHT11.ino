#include <DHT.h>

#define DHTPIN 4     // Pin D4/GPIO 4
#define DHTTYPE DHT11   

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  delay(1000);        
  dht.begin();
  delay(2000);        

  // Print the CSV Header column names first
  Serial.println("Timestamp_ms,Raw_Temp_C,Humidity_Pct,Ambient_Temp_C");
}

void loop() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();

  // If reading fails, skip this cycle instead of printing broken characters
  if (!isnan(h) && !isnan(t)) {
    float ambientC = dht.computeHeatIndex(t, h, false);
    
    // Print fields separated exactly by commas
    Serial.print(millis());
    Serial.print(",");
    Serial.print(t, 1); // 1 decimal place
    Serial.print(",");
    Serial.print(h, 1);
    Serial.print(",");
    Serial.println(ambientC, 1);
  }
  
  delay(3000); // Wait 3 seconds per log entry
}
