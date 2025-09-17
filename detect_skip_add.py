import cv2
import numpy as np
import subprocess
import time

# Path to adb executable - use your adb path
ADB_PATH = r'C:\Users\Lalitha Reddy\OneDrive\Desktop\taras\platform-tools\adb.exe'

# Template image of the "Skip Ad" button
TEMPLATE_PATH = 'skip_button.jpg'

# Confidence threshold for template matching (more sensitive now)
THRESHOLD = 0.75

def capture_screenshot():
    # Take screenshot on device
    subprocess.run([ADB_PATH, 'shell', 'screencap', '-p', '/sdcard/screen.png'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Pull screenshot to local machine
    subprocess.run([ADB_PATH, 'pull', '/sdcard/screen.png', 'screen.png'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def find_skip_button(screenshot_path, template_path):
    img_gray = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if img_gray is None or template is None:
        print("Error loading images. Check paths!")
        return None

    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= THRESHOLD)

    for pt in zip(*loc[::-1]):
        return pt, (w, h)
    return None

def click_on_device(x, y):
    # Short swipe as a tap
    subprocess.run([ADB_PATH, 'shell', 'input', 'swipe', str(x), str(y), str(x+5), str(y+5), '100'])
    print(f"Clicked at ({x}, {y})")

def main():
    print("Starting skip ad detector... (Press Ctrl+C to stop)")
    while True:
        capture_screenshot()
        result = find_skip_button('screen.png', TEMPLATE_PATH)
        if result:
            (x, y), (w, h) = result
            click_x = x + w // 2
            click_y = y + h // 2
            print(f"Skip Ad button detected at ({x}, {y}). Clicking at ({click_x}, {click_y})...")
            time.sleep(0.3)  # short delay before clicking
            click_on_device(click_x, click_y)
            time.sleep(1.5)  # short cooldown after click
        else:
            print("Skip Ad button not detected.")
            time.sleep(0.2)  # faster checking

if __name__ == '__main__':
    main()
