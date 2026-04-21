import pickle
import numpy as np
from tensorflow import keras
import time
from collections import deque
from datetime import datetime
import os

# Try to import pyserial
try:
    from serial import Serial
except ImportError:
    print("❌ PySerial is not installed!")
    print("Install it with: pip install pyserial")
    exit()

# Configuration
PORT = "COM3"
BAUD_RATE = 115200
READINGS_PER_INTERVAL = 3

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models")

# Load model
print("Loading trained GRU model...")
model_path = os.path.join(models_dir, "gru_fire_detection_model.keras")
model = keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Load scaler
print("Loading feature scaler...")
with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
print("✅ Scaler loaded successfully!")

# Labels
class_labels = {0: "SAFE", 1: "ALERT", 2: "HAZARDOUS"}

# Buffer
sensor_buffer = deque(maxlen=READINGS_PER_INTERVAL)

# Logs
log_path = os.path.join(script_dir, "predictions_log.txt")
predictions_data = []

# Serial connection
print(f"\nConnecting to {PORT}...")
try:
    ser = Serial(PORT, BAUD_RATE, timeout=1)
    print(f"✅ Connected to {PORT}")
except Exception as e:
    print(f"❌ Serial error: {e}")
    exit()

print("\n🔥 Fire Detection Started...\n")

try:
    while True:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode("latin-1").strip()
            except:
                print("❌ Decode error, skipping...")
                continue

            if not line:
                continue

            # Skip unwanted lines
            if any(
                x in line
                for x in [
                    "====",
                    "Smoke",
                    "Temperature",
                    "Humidity",
                    "Flame",
                    "Warn",
                    "Alarm",
                    "SENSOR DATA",
                ]
            ):
                continue

            # Parse data
            if "," not in line:
                continue

            try:
                values = [x.strip() for x in line.split(",")]

                if len(values) != 5:
                    continue

                smoke = float(values[1])
                temp = float(values[2])
                flame = float(values[4])

                sensor_buffer.append([smoke, temp, flame])

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Smoke={smoke:.1f}, Temp={temp:.2f}, Flame={int(flame)}"
                )

            except:
                continue

            # Prediction
            if len(sensor_buffer) == READINGS_PER_INTERVAL:
                print("\n📊 Making prediction...\n")

                readings_array = np.array(sensor_buffer)
                readings_normalized = scaler.transform(readings_array)
                readings_reshaped = readings_normalized.reshape(1, 1, 3)

                try:
                    prediction = model.predict(readings_reshaped, verbose=0)
                except Exception as e:
                    print(f"❌ Prediction error: {e}")
                    sensor_buffer.clear()
                    continue

                predicted_class = int(np.argmax(prediction[0]))
                confidence = prediction[0][predicted_class] * 100

                print("=" * 50)
                print(f"Status: {class_labels[predicted_class]}")
                print(f"Confidence: {confidence:.2f}%")
                print(f"Safe: {prediction[0][0]*100:.2f}%")
                print(f"Alert: {prediction[0][1]*100:.2f}%")
                print(f"Hazardous: {prediction[0][2]*100:.2f}%")
                print("=" * 50 + "\n")

                # Store result
                predictions_data.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": class_labels[predicted_class],
                        "confidence": round(confidence, 2),
                        "safe_prob": round(prediction[0][0] * 100, 2),
                        "alert_prob": round(prediction[0][1] * 100, 2),
                        "hazardous_prob": round(prediction[0][2] * 100, 2),
                        "avg_mq2": round(np.mean(readings_array[:, 0]), 2),
                        "avg_temp": round(np.mean(readings_array[:, 1]), 2),
                        "avg_flame": round(np.mean(readings_array[:, 2]), 2),
                    }
                )

                sensor_buffer.clear()

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")

finally:
    if predictions_data:
        print("💾 Saving logs...")

        with open(log_path, "a") as f:
            for r in predictions_data:
                f.write(str(r) + "\n")

        print(f"✅ Saved to {log_path}")

    if ser.is_open:
        ser.close()
        print("✅ Serial closed")
