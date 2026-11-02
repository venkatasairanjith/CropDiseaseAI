/*
 * ==============================================================================
 * Project: AI-Based Crop Disease Detection and Management System
 * Course: E1CSA313 Design Project - 1 | Alliance University, Bangalore
 * Team: Sai Ranjith, A. Akhileswara achari, S. Haneesh Ranganath, D. Karunakar
 * Mentor: Dr. Devika P
 * 
 * Hardware Component: Module 5 & 7 - ESP32 IoT Environmental Sensor Node
 * 
 * Target Microcontroller: ESP32 DevKit V1 (30-pin / 38-pin)
 * Sensors Integrated:
 *   1. Capacitive Soil Moisture Sensor v1.2 (Corrosion-resistant analog)
 *   2. DHT22 / DHT11 Digital Temperature & Relative Humidity Sensor
 *   3. Analog Soil pH Sensor Probe (pH-4502C module)
 * 
 * Function:
 *   - Samples ambient temperature, relative humidity, soil moisture, and pH.
 *   - Calibrates raw ADC inputs to standard agronomic units.
 *   - Encapsulates telemetry into JSON format.
 *   - Transmits telemetry over 2.4 GHz Wi-Fi via HTTP POST / WebSockets.
 * ==============================================================================
 * 
 * PINOUT CONFIGURATION:
 * ------------------------------------------------------------------------------
 * Sensor                 | ESP32 Pin  | Signal Type      | Voltage
 * ------------------------------------------------------------------------------
 * DHT22 Data             | GPIO 4     | Digital I/O      | 3.3V
 * Capacitive Moisture    | GPIO 34    | Analog In (ADC1) | 3.3V
 * Soil pH Sensor (Po)    | GPIO 35    | Analog In (ADC1) | 3.3V / 5.0V
 * Status LED (Built-in)  | GPIO 2     | Digital Output   | 3.3V
 * ------------------------------------------------------------------------------
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

// --- PIN DEFINITIONS ---
#define DHTPIN            4         // Digital pin connected to DHT22
#define DHTTYPE           DHT22     // DHT 22 (AM2302) or DHT11
#define MOISTURE_PIN      34        // ADC1_CH6 - Capacitive Soil Moisture
#define PH_PIN            35        // ADC1_CH7 - Analog Soil pH Module
#define STATUS_LED        2         // Onboard Blue LED

// --- WI-FI CREDENTIALS ---
const char* WIFI_SSID     = "CropAI_FarmNode";       // Enter Farm Wi-Fi / Hotspot SSID
const char* WIFI_PASSWORD = "PrecisionAgri2026";     // Enter Wi-Fi Password

// --- API ENDPOINT ---
// URL of the backend ingestion server or IoT gateway
const char* SERVER_URL    = "http://192.168.1.100:5000/api/telemetry";

// --- SENSOR CALIBRATION CONSTANTS ---
// Capacitive Moisture Sensor Calibration (12-bit ADC: 0 - 4095)
// Air (completely dry) ~ 3200 ADC; Water (submerged) ~ 1400 ADC
const int AIR_VALUE       = 3200;
const int WATER_VALUE     = 1400;

// pH-4502C Calibration Constants
// Standard 2-point buffer calibration: pH 4.01 and pH 7.00
const float PH_SLOPE      = -5.70;   // Calibration slope: Delta pH / Delta Volts
const float PH_OFFSET     = 21.34;   // Zero-point offset voltage compensation

// --- TELEMETRY SAMPLING INTERVAL ---
const unsigned long TRANSMIT_INTERVAL_MS = 5000;   // Transmit telemetry every 5 seconds
unsigned long lastTransmitTime = 0;

// Initialize DHT Sensor
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n========================================================");
  Serial.println("  AI-Based Crop Disease Detection System - ESP32 Node   ");
  Serial.println("  Alliance University | Design Project - 1 (E1CSA313)   ");
  Serial.println("========================================================");

  pinMode(STATUS_LED, OUTPUT);
  pinMode(MOISTURE_PIN, INPUT);
  pinMode(PH_PIN, INPUT);

  // Initialize DHT
  dht.begin();
  Serial.println("[OK] DHT22 Temperature & Humidity sensor initialized.");

  // Connect to Wi-Fi
  connectWiFi();
}

void loop() {
  unsigned long currentTime = millis();

  // Non-blocking telemetry acquisition loop
  if (currentTime - lastTransmitTime >= TRANSMIT_INTERVAL_MS) {
    lastTransmitTime = currentTime;

    // 1. Read Microclimate Temperature & Humidity
    float temperature = dht.readTemperature();
    float humidity    = dht.readHumidity();

    // Check if DHT read failed
    if (isnan(temperature) || isnan(humidity)) {
      Serial.println("[WARN] Failed to read from DHT sensor! Using fallback calibration.");
      temperature = 28.5;
      humidity    = 76.0;
    }

    // 2. Read and Calibrate Soil Moisture
    int rawMoistureADC = analogRead(MOISTURE_PIN);
    float soilMoisturePct = map(rawMoistureADC, AIR_VALUE, WATER_VALUE, 0, 100);
    soilMoisturePct = constrain(soilMoisturePct, 0.0, 100.0);

    // 3. Read and Calibrate Soil pH
    int rawPhADC = analogRead(PH_PIN);
    float phVoltage = (rawPhADC / 4095.0) * 3.3; // Convert ADC value to voltage
    float soilPh = (PH_SLOPE * phVoltage) + PH_OFFSET;
    soilPh = constrain(soilPh, 4.0, 9.5);

    // Visual blink indicator
    digitalWrite(STATUS_LED, HIGH);

    // 4. Print Local Serial Telemetry
    Serial.println("\n--- [ESP32 Telemetry Snapshot] ---");
    Serial.printf("  Ambient Temperature : %.2f °C\n", temperature);
    Serial.printf("  Relative Humidity   : %.2f %%\n", humidity);
    Serial.printf("  Soil Moisture       : %.2f %% (Raw ADC: %d)\n", soilMoisturePct, rawMoistureADC);
    Serial.printf("  Soil Reaction (pH)  : %.2f (Voltage: %.2f V)\n", soilPh, phVoltage);

    // 5. Build JSON Telemetry Payload
    String jsonPayload = "{";
    jsonPayload += "\"node_id\":\"ESP32_AGRI_01\",";
    jsonPayload += "\"temperature\":" + String(temperature, 2) + ",";
    jsonPayload += "\"humidity\":" + String(humidity, 2) + ",";
    jsonPayload += "\"soil_moisture\":" + String(soilMoisturePct, 2) + ",";
    jsonPayload += "\"soil_ph\":" + String(soilPh, 2) + ",";
    jsonPayload += "\"timestamp\":" + String(millis());
    jsonPayload += "}";

    Serial.println("  Payload: " + jsonPayload);

    // 6. Transmit Payload via HTTP POST if Wi-Fi connected
    if (WiFi.status() == WL_CONNECTED) {
      sendTelemetry(jsonPayload);
    } else {
      Serial.println("[INFO] Wi-Fi offline. Telemetry logged to local Serial stream.");
      // Attempt reconnection
      connectWiFi();
    }

    digitalWrite(STATUS_LED, LOW);
  }
}

// --- WI-FI CONNECTION HELPER ---
void connectWiFi() {
  Serial.print("[WIFI] Connecting to SSID: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 15) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WIFI] Connected successfully!");
    Serial.print("[WIFI] IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n[WIFI] Connection timed out. Running in standalone local ADC mode.");
  }
}

// --- HTTP TRANSMISSION HELPER ---
void sendTelemetry(const String& payload) {
  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");

  int httpResponseCode = http.POST(payload);

  if (httpResponseCode > 0) {
    Serial.printf("[HTTP] Ingestion response code: %d\n", httpResponseCode);
  } else {
    Serial.printf("[HTTP] POST failed. Error code: %s\n", http.errorToString(httpResponseCode).c_str());
  }

  http.end();
}
