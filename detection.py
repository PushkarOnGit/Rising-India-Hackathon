import cv2
from ultralytics import YOLO
import os
import time

# ==============================
# 🔧 LOAD TRAINED MODEL
# ==============================

# Use correct model path - change from fire_smoke_best to train
model_path = "runs/detect/train/weights/best.pt"

if not os.path.exists(model_path):
    print(f"❌ Model not found at {model_path}")
    print("Available model files:")
    if os.path.exists("runs/detect/train/weights/"):
        print(f"  - {model_path}")
    else:
        print("  No trained models found in runs/detect/")
    exit()

model = YOLO(model_path)

# Get class names dynamically
class_names = model.names

# ==============================
# 📁 CREATE ALERT FOLDER
# ==============================

ALERT_PATH = "alerts/images"
os.makedirs(ALERT_PATH, exist_ok=True)

# ==============================
# ⚙️ PARAMETERS
# ==============================

CONF_THRESHOLD = 0.5
SAVE_INTERVAL = 5  # seconds

last_saved_time = 0

# ==============================
# 📷 OPEN CAMERA
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("🔥 Fire Detection Started (Press ESC to exit)")

# ==============================
# 🚀 MAIN LOOP
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ==============================
    # 🧠 RUN DETECTION
    # ==============================

    results = model(frame, conf=CONF_THRESHOLD)

    fire_detected = False
    smoke_detected = False

    # ==============================
    # 🔍 PROCESS RESULTS
    # ==============================

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = class_names[cls]

            if label == "fire":
                fire_detected = True
            elif label == "smoke":
                smoke_detected = True

    # ==============================
    # 🎯 STATUS LOGIC
    # ==============================

    if fire_detected and smoke_detected:
        status = "🔥 FIRE + SMOKE"
        color = (0, 0, 255)

    elif fire_detected:
        status = "🔥 FIRE"
        color = (0, 0, 255)

    elif smoke_detected:
        status = "💨 SMOKE"
        color = (0, 255, 255)

    else:
        status = "✅ NORMAL"
        color = (0, 255, 0)

    # ==============================
    # 🖼️ DRAW RESULTS
    # ==============================

    annotated_frame = results[0].plot()

    cv2.putText(
        annotated_frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3
    )

    # ==============================
    # 💾 SAVE ALERT IMAGE
    # ==============================

    current_time = time.time()

    if (fire_detected or smoke_detected) and (
        current_time - last_saved_time > SAVE_INTERVAL
    ):
        filename = f"{ALERT_PATH}/alert_{int(current_time)}.jpg"
        cv2.imwrite(filename, annotated_frame)

        print(f"🚨 ALERT: {status}")
        print(f"📸 Saved: {filename}")

        last_saved_time = current_time

    # ==============================
    # 🖥️ DISPLAY
    # ==============================

    cv2.imshow("🔥 Fire Detection System", annotated_frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==============================
# 🧹 CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()
