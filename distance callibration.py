import cv2
import numpy as np

# ----------------- CALIBRATION SETUP -----------------
# 0 for webcam, or phone URL (e.g. "http://192.168.1.15:8080/video")
CAMERA_SOURCE = 1

# Test distance: EXACTLY 1.00 meter (100 cm) from the camera lens
KNOWN_DISTANCE_M = 1.0

# Bottle measurement:
# Radius = 3.7 cm -> Diameter = 2 * 3.7 cm = 7.4 cm = 0.074 meters
KNOWN_WIDTH_M = 0.211

FRAME_W = 640
FRAME_H = 480
# -----------------------------------------------------

cap = cv2.VideoCapture(CAMERA_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

points = []
calibrated_focal_length = None
last_pixel_width = 0.0


def click_event(event, x, y, flags, param):
    global points, calibrated_focal_length, last_pixel_width
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"[+] Point recorded at: ({x}, {y})")

        # Once 2 points are selected (left edge & right edge)
        if len(points) == 2:
            pt1, pt2 = points[0], points[1]
            last_pixel_width = float(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))

            # Pinhole formula: f = (pixel_width * distance) / real_width
            calibrated_focal_length = (last_pixel_width * KNOWN_DISTANCE_M) / KNOWN_WIDTH_M

            print("\n" + "=" * 55)
            print("                 CALIBRATION SUCCESSFUL")
            print("=" * 55)
            print(f"Target Object Diameter: {KNOWN_WIDTH_M * 100:.1f} cm ({KNOWN_WIDTH_M} m)")
            print(f"Measured Pixel Width  : {last_pixel_width:.1f} px")
            print(f"CALIBRATED FOCAL_LENGTH : {calibrated_focal_length:.1f}")
            print("=" * 55)
            print(">> Paste this FOCAL_LENGTH value into your demo.py:")
            print(f"FOCAL_LENGTH = {calibrated_focal_length:.1f}")
            print("=" * 55 + "\n")


cv2.namedWindow("Focal Length Calibration")
cv2.setMouseCallback("Focal Length Calibration", click_event)

print("\n-----------------------------------------------------------")
print(f"1. Place your bottle standing upright EXACTLY {KNOWN_DISTANCE_M:.2f} m from the camera.")
print("2. Left-click the LEFT outer edge of the bottle.")
print("3. Left-click the RIGHT outer edge of the bottle.")
print("4. Press 'r' to reset points and measure again.")
print("5. Press 'q' to quit.")
print("-----------------------------------------------------------\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))

    # Crosshair alignment guide
    cv2.line(frame, (FRAME_W // 2, 0), (FRAME_W // 2, FRAME_H), (40, 40, 40), 1)
    cv2.line(frame, (0, FRAME_H // 2), (FRAME_W, FRAME_H // 2), (40, 40, 40), 1)

    # Draw clicked points
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    # Draw line between selected edges
    if len(points) == 2:
        cv2.line(frame, points[0], points[1], (0, 255, 0), 2)

    # Overlay instructions / results
    if calibrated_focal_length is None:
        cv2.putText(frame, f"Bottle at {KNOWN_DISTANCE_M}m: Click Left & Right edges", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Focal Length: {calibrated_focal_length:.1f} (Press 'r' to redo)", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Width: {last_pixel_width:.1f}px", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Focal Length Calibration", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        points = []
        calibrated_focal_length = None
        print("\n[*] Points reset. Ready for new measurement.")

cap.release()
cv2.destroyAllWindows()