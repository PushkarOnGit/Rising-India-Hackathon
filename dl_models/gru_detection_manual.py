import serial
import time

# =========================
# CONFIG
# =========================
PORT = "COM3"  # बदल: Windows -> COM5, Linux -> /dev/ttyUSB0
BAUD = 115200

# =========================
# CONNECT SERIAL
# =========================
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # ESP32 reset time
    print(f"Connected to {PORT} at {BAUD} baud\n")
except:
    print("Failed to connect. Check COM port.")
    exit()

# =========================
# READ LOOP
# =========================
while True:
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        # Ignore empty lines
        if not line:
            continue

        # Ignore debug text (only allow lines starting with number)
        if not line[0].isdigit():
            continue

        # Expected: time,smoke,temp,hum,flame
        parts = line.split(",")

        if len(parts) != 5:
            continue

        # Parse values
        timestamp = int(parts[0])
        smoke = int(parts[1])
        temp = float(parts[2])
        hum = float(parts[3])
        flame = int(parts[4])

        # Output (you can replace this with model input later)
        print("------ SENSOR DATA ------")
        print(f"Time       : {timestamp}")
        print(f"Smoke      : {smoke}")
        print(f"Temperature: {temp} °C")
        print(f"Humidity   : {hum} %")
        print(f"Flame      : {'YES' if flame else 'NO'}")
        print("--------------------------\n")

    except KeyboardInterrupt:
        print("\nStopped by user")
        break

    except Exception as e:
        print("Error:", e)
