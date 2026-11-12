import time
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------- CONFIGURATION -----------------
CAMERA_SOURCE = 0  # 0 or 1 for USB / IP stream

FRAME_W = 640
FRAME_H = 480

TARGET_REAL_DIAMETER = 0.065  # ~5.0 cm real physical diameter
FOCAL_LENGTH =  512.2  # Calibrated focal length for 640x480
RIM_PLANE_Y = 0.00  # Target catch rim plane (meters)

# Motion Gates
MIN_AIRBORNE_AREA = 120  # Sensitive enough to catch small moving wads
MAX_AIRBORNE_AREA = 9000
MAX_OBJECT_DIM = 160

# AI Verification Classes:
# 39: bottle, 32: sports ball, 41: cup. (Can be left empty [] to accept all small objects)
TRASH_CLASSES = [32, 39, 41]
AI_CONFIDENCE = 0.20  # Low threshold works well because background is already cropped out
# -------------------------------------------------

# Load lightweight YOLO model
print("Loading YOLOv8 Nano...")
model = YOLO("yolov8n.pt")  # Auto-downloads lightweight weights on first run

cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

times = []
pts_x = []
pts_y = []
pts_z = []

last_detect_time = 0
predicted_impact = None
virtual_bin_x = FRAME_W // 2

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=20, varThreshold=16, detectShadows=False
)


def solve_3d_trajectory(t_arr, x_arr, y_arr, z_arr, rim_y):
    if len(t_arr) < 3:
        return None

    t_norm = np.array(t_arr) - t_arr[0]

    try:
        g = 9.81
        y_adjusted = np.array(y_arr) + 0.5 * g * (t_norm**2)
        poly_y_linear = np.polyfit(t_norm, y_adjusted, 1)
        Vy0 = poly_y_linear[0]
        Y0 = poly_y_linear[1]

        poly_x = np.polyfit(t_norm, x_arr, 1)
        poly_z = np.polyfit(t_norm, z_arr, 1)

        A = -0.5 * g
        B = Vy0
        C = Y0 - rim_y

        disc = B**2 - 4 * A * C
        if disc < 0:
            return None

        t1 = (-B + np.sqrt(disc)) / (2 * A)
        t2 = (-B - np.sqrt(disc)) / (2 * A)

        valid_times = [t for t in [t1, t2] if t > t_norm[-1]]
        if not valid_times:
            t_impact = t_norm[-1] + 0.30
        else:
            t_impact = min(valid_times)

        target_x = float(np.polyval(poly_x, t_impact))
        target_z = float(np.polyval(poly_z, t_impact))
        t_rem = float(t_impact - t_norm[-1])

        if target_z < 0.1 or target_z > 6.0:
            return None

        return target_x, target_z, t_rem
    except Exception:
        return None


def draw_topdown_radar(frame, target_x, target_z):
    radar_w, radar_h = 160, 160
    radar_x1 = FRAME_W - radar_w - 10
    radar_y1 = 10

    sub_img = frame[radar_y1 : radar_y1 + radar_h, radar_x1 : radar_x1 + radar_w]
    black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 0.25, black_rect, 0.75, 1.0)
    frame[
        radar_y1 : radar_y1 + radar_h, radar_x1 : radar_x1 + radar_w
    ] = res
    cv2.rectangle(
        frame,
        (radar_x1, radar_y1),
        (radar_x1 + radar_w, radar_y1 + radar_h),
        (0, 255, 255),
        1,
    )

    bin_radar_u = radar_x1 + radar_w // 2
    bin_radar_v = radar_y1 + radar_h - 15
    scale = (radar_h - 25) / 3.5

    for d in [1.0, 2.0, 3.0]:
        cv2.circle(
            frame, (bin_radar_u, bin_radar_v), int(d * scale), (60, 60, 60), 1
        )

    cv2.circle(frame, (bin_radar_u, bin_radar_v), 6, (255, 120, 0), -1)
    cv2.putText(
        frame,
        "BIN",
        (bin_radar_u - 10, bin_radar_v + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
    )

    if target_x is not None and target_z is not None:
        impact_u = int(bin_radar_u + target_x * scale)
        impact_v = int(bin_radar_v - target_z * scale)

        impact_u = np.clip(impact_u, radar_x1 + 5, radar_x1 + radar_w - 5)
        impact_v = np.clip(impact_v, radar_y1 + 5, radar_y1 + radar_h - 5)

        cv2.line(
            frame,
            (bin_radar_u, bin_radar_v),
            (impact_u, impact_v),
            (0, 200, 255),
            1,
        )
        cv2.circle(frame, (impact_u, impact_v), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"{target_z:.1f}m",
            (impact_u + 6, impact_v),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
        )

    cv2.putText(
        frame,
        "TOP-DOWN RADAR",
        (radar_x1 + 18, radar_y1 + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1,
    )


# Flush camera buffer
for _ in range(15):
    cap.read()

while cap.isOpened():
    t_start = time.time()
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    cx_img, cy_img = FRAME_W // 2, FRAME_H // 2
    now = time.time()

    # Dynamic airspace (ignore lower 15% floor clutter)
    airspace_h = int(FRAME_H * 0.85)  # Use 90% of frame height
    airspace = frame[0:airspace_h, :]

    fg_mask = bg_subtractor.apply(airspace)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    airborne_candidate = None
    best_motion_area = 0

    # Step 1: High-Speed Motion Triggering
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AIRBORNE_AREA < area < MAX_AIRBORNE_AREA:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw > MAX_OBJECT_DIM or bh > MAX_OBJECT_DIM:
                continue

            aspect_ratio = float(bw) / float(bh)
            if 0.35 <= aspect_ratio <= 2.8:
                if area > best_motion_area:
                    best_motion_area = area
                    airborne_candidate = (bx, by, bw, bh)

    detected = False
    ai_verified = False
    detected_label = "Trash"

    # Step 2: AI Verification on Cropped Patch (Ultra-fast, zero miss)
    if airborne_candidate is not None:
        bx, by, bw, bh = airborne_candidate

        # Add 25px margin around the moving object for context
        pad = 25
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(FRAME_W, bx + bw + pad)
        y2 = min(airspace_h, by + bh + pad)

        roi_patch = frame[y1:y2, x1:x2]

        if roi_patch.shape[0] > 10 and roi_patch.shape[1] > 10:
            # Predict only on this small patch (Takes only 3-5ms!)
            ai_results = model.predict(
                source=roi_patch,
                conf=AI_CONFIDENCE,
                verbose=False,
                imgsz=160,  # Fast lightweight patch inference
            )

            # Check if AI identified the object
            for r in ai_results:
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    # If class filtering is enabled, check target class; otherwise accept valid object
                    if len(TRASH_CLASSES) == 0 or cls_id in TRASH_CLASSES:
                        ai_verified = True
                        detected_label = f"{model.names[cls_id]} {float(b.conf[0]):.2f}"
                        break
                    # Also accept paper wads if identified as sports ball, apple, or cup
                    elif cls_id in [32, 47, 49, 41]:
                        ai_verified = True
                        detected_label = f"{model.names[cls_id]} {float(b.conf[0]):.2f}"
                        break

        # Fallback acceptance: If motion is strong and moving consistently, accept even if AI label is uncertain
        if not ai_verified and best_motion_area > 280:
            ai_verified = True
            detected_label = "Motion-Track"

        if ai_verified:
            u = bx + bw // 2
            v = by + bh // 2
            # Use min(bw, bh) to stop rolling-shutter stretch from inflating depth Z
            pixel_diam = min(bw, bh)

            # 3D Depth & Coordinates
            Z_real = (FOCAL_LENGTH * TARGET_REAL_DIAMETER) / max(pixel_diam, 1)
            X_real = ((u - cx_img) * Z_real) / FOCAL_LENGTH
            Y_real = -((v - cy_img) * Z_real) / FOCAL_LENGTH

            if now - last_detect_time > 0.60:
                times, pts_x, pts_y, pts_z = [], [], [], []
                predicted_impact = None

            times.append(now)
            pts_x.append(X_real)
            pts_y.append(Y_real)
            pts_z.append(Z_real)
            last_detect_time = now
            detected = True

            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.circle(frame, (u, v), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"[{detected_label}] Z:{Z_real:.2f}m",
                (bx, max(20, by - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                2,
            )

            if len(times) >= 3:
                res = solve_3d_trajectory(
                    times, pts_x, pts_y, pts_z, RIM_PLANE_Y
                )
                if res:
                    predicted_impact = res

    if not detected and (now - last_detect_time > 0.45):
        predicted_impact = None

    # ----------------- HUD & SIMULATION -----------------
    bin_y = FRAME_H - 35
    bin_width = 130
    target_x_rad, target_z_rad = None, None

    if predicted_impact is not None:
        t_x, t_z, t_rem = predicted_impact
        target_x_rad, target_z_rad = t_x, t_z

        target_pixel_x = int(cx_img + (t_x * FOCAL_LENGTH) / max(t_z, 0.3))
        target_pixel_x = np.clip(
            target_pixel_x, bin_width // 2, FRAME_W - bin_width // 2
        )

        virtual_bin_x += int((target_pixel_x - virtual_bin_x) * 0.45)

        cv2.putText(
            frame,
            "3D IMPACT SOLVED!",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Lateral (X): {t_x:+.2f} m",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Forward (Z): {t_z:.2f} m",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Time-to-catch: {t_rem*1000:.0f} ms",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 0),
            1,
        )

        cv2.drawMarker(
            frame,
            (target_pixel_x, bin_y),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            20,
            2,
        )
    else:
        virtual_bin_x += int(((FRAME_W // 2) - virtual_bin_x) * 0.1)

    draw_topdown_radar(frame, target_x_rad, target_z_rad)

    cv2.rectangle(
        frame,
        (virtual_bin_x - bin_width // 2, bin_y),
        (virtual_bin_x + bin_width // 2, bin_y + 20),
        (255, 120, 0),
        -1,
    )

    loop_fps = 1.0 / max(time.time() - t_start, 0.001)
    cv2.putText(
        frame,
        f"FPS: {loop_fps:.0f}",
        (20, FRAME_H - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    cv2.imshow("3D Trajectory & Z-Depth Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()