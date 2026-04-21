import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import os

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models")

# Load trained model
print("Loading trained GRU model...")
model_path = os.path.join(models_dir, "gru_fire_detection_model.keras")
if not os.path.exists(model_path):
    print(f"❌ Model not found at {model_path}")
    exit()
model = keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Load scaler
print("Loading feature scaler...")
scaler_path = os.path.join(models_dir, "scaler.pkl")
if not os.path.exists(scaler_path):
    print(f"❌ Scaler not found at {scaler_path}")
    exit()
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
print("✅ Scaler loaded successfully!\n")

# Class labels
class_labels = {0: "🟢 SAFE", 1: "🟡 ALERT", 2: "🔴 HAZARDOUS"}

# Create predictions log file
log_path = os.path.join(script_dir, "predictions_log.txt")
predictions_data = []

print("=" * 60)
print("🔥 Fire Detection System - Manual Input")
print("=" * 60)
print("Enter 3 sensor readings (MQ2, Temperature, Flame)\n")

try:
    while True:
        readings = []

        print("\n" + "-" * 60)
        print("Enter 3 sensor readings:")
        print("-" * 60)

        for i in range(3):
            while True:
                try:
                    if i == 0:
                        value = float(input(f"Reading {i+1} - MQ2/Smoke (0-1200): "))
                        if 0 <= value <= 1200:
                            readings.append(value)
                            break
                        else:
                            print("❌ MQ2 must be between 0-1200!")
                    elif i == 1:
                        value = float(
                            input(f"Reading {i+1} - Temperature (°C, 0-100): ")
                        )
                        if 0 <= value <= 100:
                            readings.append(value)
                            break
                        else:
                            print("❌ Temperature must be between 0-100!")
                    elif i == 2:
                        value = float(input(f"Reading {i+1} - Flame (0=No, 1=Yes): "))
                        if value in [0, 1]:
                            readings.append(value)
                            break
                        else:
                            print("❌ Flame must be 0 or 1!")
                except ValueError:
                    print("❌ Invalid input! Please enter a number.")

        # Display received readings
        print(f"\n✅ Received readings:")
        print(
            f"   MQ2: {readings[0]:.1f}, Temperature: {readings[1]:.1f}°C, Flame: {readings[2]}"
        )

        # Convert to numpy array
        readings_array = np.array([readings])

        # Normalize features
        readings_normalized = scaler.transform(readings_array)

        # Reshape for GRU: (batch_size=1, sequence_length=1, features=3)
        readings_reshaped = readings_normalized.reshape(1, 1, 3)

        # Get prediction
        prediction = model.predict(readings_reshaped, verbose=0)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = prediction[0][predicted_class] * 100

        # Display results
        print("\n" + "=" * 60)
        print(f"⏰ Prediction Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Status: {class_labels[predicted_class]}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"\nAll probabilities:")
        print(f"  🟢 Safe:      {prediction[0][0]*100:.2f}%")
        print(f"  🟡 Alert:     {prediction[0][1]*100:.2f}%")
        print(f"  🔴 Hazardous: {prediction[0][2]*100:.2f}%")
        print("=" * 60)

        # Store prediction data
        prediction_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": class_labels[predicted_class].split()[
                1
            ],  # Just the status without emoji
            "confidence": round(confidence, 2),
            "safe_prob": round(prediction[0][0] * 100, 2),
            "alert_prob": round(prediction[0][1] * 100, 2),
            "hazardous_prob": round(prediction[0][2] * 100, 2),
            "mq2": round(readings[0], 2),
            "temp": round(readings[1], 2),
            "flame": int(readings[2]),
        }
        predictions_data.append(prediction_record)

        # Ask if user wants to continue
        while True:
            cont = input("\n🔁 Make another prediction? (yes/no): ").lower().strip()
            if cont in ["yes", "y"]:
                break
            elif cont in ["no", "n"]:
                raise KeyboardInterrupt
            else:
                print("❌ Please enter 'yes' or 'no'")

except KeyboardInterrupt:
    print("\n\n🛑 Stopping detection system...")

except Exception as e:
    print(f"\n❌ Error occurred: {e}")

finally:
    # Save predictions to file
    if predictions_data:
        print(f"\n💾 Saving {len(predictions_data)} predictions to file...")

        # Save as text log
        with open(log_path, "a") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(
                f"Manual Input Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write("=" * 80 + "\n")
            for record in predictions_data:
                f.write(f"[{record['timestamp']}] Status: {record['status']} | ")
                f.write(f"Confidence: {record['confidence']}% | ")
                f.write(
                    f"Probs(Safe:{record['safe_prob']}%, Alert:{record['alert_prob']}%, Hazard:{record['hazardous_prob']}%)\n"
                )
                f.write(
                    f"  MQ2: {record['mq2']}, Temp: {record['temp']}°C, Flame: {record['flame']}\n"
                )

        print(f"✅ Text log saved to: {log_path}")

        # Save as CSV for analysis
        import csv

        csv_path = os.path.join(script_dir, "predictions_data.csv")
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=predictions_data[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(predictions_data)

        print(f"✅ CSV data saved to: {csv_path}")

    print("\n✅ Detection system stopped.")
