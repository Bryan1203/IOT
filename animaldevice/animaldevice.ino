#include "Adafruit_GPS.h"
#include "PulseOximeter.h"
#include "MQ135.h"

Adafruit_GPS GPS;
PulseOximeter pox;
MQ135 gasSensor = MQ135(A2);

void setup() {
  //vital
  pinMode(2, INPUT);
  pinMode(3, OUTPUT);

  //temperature
  pinMode(A1, INPUT);

  //air
  pinMode(A2, INPUT);
  pinMode(3, OUTPUT);
  pinMode(8, INPUT);

  Serial.begin(9600);
}
void loop() {
  //GPS
  if (Serial.available() > 0) {	
    string gpsResult = GPS.read();
    Serial.println(gpsResult);
  } else {
    Serial.println("No data");
  }
  
  //Vitals
  if (pox.begin()) {
    string spO2_res = pox.getSpO2();
    string hb_res = pox.getHeartRate();
    Serial.println("OxygenOxigen percentage: " + spO2_res + "; Heart rate: " + hb_res);
    digitalWrite(3, HIGH);
  } else {
    Serial.println("No vitals data");
    digitalWrite(3, LOW);
  }
  //temperature
  int temp_adc_val;
  float temp_val;
  temp_adc_val = analogRead(A1);	/* Read Temperature */
  temp_val = (temp_adc_val * 4.88);	/* Convert adc value to equivalent voltage */
  temp_val = (temp_val/10);	/* LM35 gives output of 10mv/°C */
  //float temp = analogRead(A1) / 1023.0 * 5.0 * 100.0;  // temperature = analog voltage * 100
  Serial.println("temperature: " + to_string(temp_val));

  //air
  float ppm = gasSensor.getPPM();
  Serial.println("ppm: " + to_string(ppm));

  int digitalResult = digitalRead(8);
  if (digitalResult == HIGH) digitalWrite(3, HIGH);
  else digitalWrite(3, LOW);

  delay(1000);
}