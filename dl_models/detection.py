import cv2
from ultralytics import YOLO
import os
import time

# loading model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "runs/detect/train-2/weights/best.pt")

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    exit()

model = YOLO(model_path)
class_names = model.names

# alert folder

ALERT_PATH = "alerts/images"
os.makedirs(ALERT_PATH, exist_ok=True)

# parameters

CONF_THRESHOLD = 0.5
COOLDOWN = 10  # seconds (optional safety)

last_saved_time = 0

# state tracking
prev_fire_state = False
prev_smoke_state = False

# open camera

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Fire Detection Started (Press ESC to exit)")

# loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # detection running
    results = model(frame, conf=CONF_THRESHOLD)

    fire_detected = False
    smoke_detected = False

    # processes results

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = class_names[cls]

            if label == "fire":
                fire_detected = True
            elif label == "smoke":
                smoke_detected = True

    if fire_detected and smoke_detected:
        status = "FIRE + SMOKE"
        color = (0, 0, 255)

    elif fire_detected:
        status = "FIRE"
        color = (0, 0, 255)

    elif smoke_detected:
        status = "💨 SMOKE"
        color = (0, 255, 255)

    else:
        status = "NORMAL"
        color = (0, 255, 0)

    # drawing results

    annotated_frame = results[0].plot()

    cv2.putText(
        annotated_frame,
        status,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3,
    )

    # Alert Logic

    current_time = time.time()

    # Detect NEW event (rising edge)
    new_fire_event = fire_detected and not prev_fire_state
    new_smoke_event = smoke_detected and not prev_smoke_state

    if (new_fire_event or new_smoke_event) and (
        current_time - last_saved_time > COOLDOWN
    ):
        filename = f"{ALERT_PATH}/alert_{int(current_time)}.jpg"
        cv2.imwrite(filename, annotated_frame)

        print(f"🚨 NEW ALERT: {status}")
        print(f"📸 Saved: {filename}")

        last_saved_time = current_time

    # updating State

    prev_fire_state = fire_detected
    prev_smoke_state = smoke_detected

    # display result

    cv2.imshow("Fire Detection System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# memory cleanup

cap.release()
cv2.destroyAllWindows()
