import cv2
import numpy as np

def get_color_name(h, s, v):
    if s < 50 and v > 200:
        return "WHITE"
    elif s < 50 and v < 50:
        return "BLACK"
    elif h < 10 or h > 160:
        return "RED"
    elif 10 < h < 25:
        return "ORANGE"
    elif 25 < h < 35:
        return "YELLOW"
    elif 35 < h < 85:
        return "GREEN"
    elif 85 < h < 125:
        return "BLUE"
    elif 125 < h < 150:
        return "PURPLE"
    else:
        return "UNKNOWN"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    height, width, _ = frame.shape
    x, y = width // 2 - 20, height // 2 - 20
    roi = frame[y:y+40, x:x+40]

 
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv_roi, axis=(0, 1)).astype(int)
    h, s, v = avg_hsv

    color_name = get_color_name(h, s, v)


    cv2.rectangle(frame, (x, y), (x+40, y+40), (255, 255, 255), 2)
    cv2.putText(frame, f"Color: {color_name}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
