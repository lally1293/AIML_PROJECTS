import cv2
from ultralytics import YOLO
import time

# Load the trained YOLOv8 fight detection model
model = YOLO("fight_detection.pt")  # Use your actual model name here

# Load video file (or use 0 for webcam)
cap = cv2.VideoCapture("fight1.mp4")

# Detection threshold
CONFIDENCE_THRESHOLD = 0.4  # You can lower this if detection is missing

print("📹 Fight detection started. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame if too large (optional)
    # frame = cv2.resize(frame, (640, 480))

    # Run detection
    results = model(frame)[0]

    fight_detected = False

    for r in results.boxes:
        conf = r.conf[0].item()
        cls_id = int(r.cls[0].item())
        label = model.names[cls_id]

        if conf > CONFIDENCE_THRESHOLD:
            if "fight" in label.lower():  # if label is "fight" or similar
                fight_detected = True

            # Draw bounding box
            xyxy = r.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if fight_detected:
        cv2.putText(frame, "⚠ FIGHT DETECTED!", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Fight Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
