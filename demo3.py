import time
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------- CONFIGURATION -----------------
CAMERA_SOURCE = 1  # 0 or 1 for USB / IP stream

FRAME_W = 640
FRAME_H = 480

TARGET_REAL_DIAMETER = 0.050  # ~5.0 cm real physical diameter
FOCAL_LENGTH = 512.2          # Calibrated focal length for 640x480
RIM_PLANE_Z = 0.0             # Catch rim height relative to lens (0.0 = lens level)

# Motion Gates
MIN_AIRBORNE_AREA = 80
MAX_AIRBORNE_AREA = 12000
MAX_OBJECT_DIM = 220

# Guardrails
BORDER_MARGIN = 4
MIN_PROXIMITY_Z = 0.20        # Discard points closer than 20 cm
MAX_PROXIMITY_Z = 5.50
MAX_DELTA_Z_SPEED = 14.0      # Max realistic vertical velocity (m/s)
PREDICTION_FREEZE_Z = 0.35    # Freeze prediction below 35 cm height

# AI Verification Classes: 39: bottle, 32: sports ball, 41: cup ([] accepts all)
TRASH_CLASSES = [32, 39, 41]
AI_CONFIDENCE = 0.15
# -------------------------------------------------

print("Loading YOLOv8 Nano...")
model = YOLO("yolov8n.pt")

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

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=15, varThreshold=14, detectShadows=False
)


def solve_3d_trajectory_upward(t_arr, x_arr, y_arr, z_arr, rim_z=0.0):
    if len(t_arr) < 3:
        return None

    t_norm = np.array(t_arr) - t_arr[0]
    t_curr = t_norm[-1]

    try:
        g = 9.81
        # Model: Z(t) = -0.5 * g * t^2 + Vz0 * t + Z0
        z_adj = np.array(z_arr) + 0.5 * g * (t_norm ** 2)
        poly_z = np.polyfit(t_norm, z_adj, 1)
        Vz0 = poly_z[0]
        Z0 = poly_z[1]

        # Residual check on vertical gravity model
        z_pred = poly_z[0] * t_norm + poly_z[1]
        z_error = np.mean(np.abs(z_adj - z_pred))
        if z_error > 0.35:
            return None

        # Constant velocity assumption on ground plane axes
        poly_x = np.polyfit(t_norm, x_arr, 1)
        poly_y = np.polyfit(t_norm, y_arr, 1)

        # Solve: -0.5 * g * t^2 + Vz0 * t + (Z0 - rim_z) = 0
        A = -0.5 * g
        B = Vz0
        C = Z0 - rim_z

        disc = B**2 - 4 * A * C
        if disc <= 0:
            return None

        sqrt_disc = np.sqrt(disc)
        # With A = -4.905, the forward-in-time landing root is (-B - sqrt_disc) / (2A)
        t_impact = (-B - sqrt_disc) / (2.0 * A)

        # Must land in the future within a realistic flight window
        if t_impact <= t_curr or (t_impact - t_curr) > 2.5:
            return None

        target_x = float(np.polyval(poly_x, t_impact))
        target_y = float(np.polyval(poly_y, t_impact))
        t_rem = float(t_impact - t_curr)

        # Reject impossible wild horizontal offsets
        if np.sqrt(target_x**2 + target_y**2) > 2.5:
            return None

        return target_x, target_y, t_rem
    except Exception:
        return None


def draw_upward_radar(frame, target_x, target_y):
    radar_w, radar_h = 160, 160
    radar_x1 = FRAME_W - radar_w - 10
    radar_y1 = 10

    sub_img = frame[radar_y1:radar_y1 + radar_h, radar_x1:radar_x1 + radar_w]
    black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 0.25, black_rect, 0.75, 1.0)
    frame[radar_y1:radar_y1 + radar_h, radar_x1:radar_x1 + radar_w] = res
    cv2.rectangle(frame, (radar_x1, radar_y1), (radar_x1 + radar_w, radar_y1 + radar_h), (0, 255, 255), 1)

    center_u = radar_x1 + radar_w // 2
    center_v = radar_y1 + radar_h // 2
    scale = (radar_h // 2 - 10) / 2.0  # 2.0m radius representation

    for d in [0.5, 1.0, 1.5]:
        cv2.circle(frame, (center_u, center_v), int(d * scale), (60, 60, 60), 1)

    cv2.circle(frame, (center_u, center_v), 5, (255, 120, 0), -1)
    cv2.putText(frame, "ROBOT", (center_u - 15, center_v + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

    if target_x is not None and target_y is not None:
        impact_u = int(center_u + target_x * scale)
        impact_v = int(center_v - target_y * scale)

        impact_u = np.clip(impact_u, radar_x1 + 5, radar_x1 + radar_w - 5)
        impact_v = np.clip(impact_v, radar_y1 + 5, radar_y1 + radar_h - 5)

        cv2.line(frame, (center_u, center_v), (impact_u, impact_v), (0, 200, 255), 1)
        cv2.circle(frame, (impact_u, impact_v), 5, (0, 0, 255), -1)
        dist = np.sqrt(target_x**2 + target_y**2)
        cv2.putText(frame, f"{dist:.2f}m", (impact_u + 6, impact_v),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    cv2.putText(frame, "OVERHEAD RADAR", (radar_x1 + 18, radar_y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)


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

    # 0% crop: full sensor view for upward orientation
    airspace = frame

    fg_mask = bg_subtractor.apply(airspace)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    airborne_candidate = None
    best_motion_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AIRBORNE_AREA < area < MAX_AIRBORNE_AREA:
            bx, by, bw, bh = cv2.boundingRect(cnt)

            # Edge guard across full sensor frame
            if bx <= BORDER_MARGIN or by <= BORDER_MARGIN or \
               (bx + bw) >= (FRAME_W - BORDER_MARGIN) or (by + bh) >= (FRAME_H - BORDER_MARGIN):
                continue

            if bw > MAX_OBJECT_DIM or bh > MAX_OBJECT_DIM:
                continue

            aspect_ratio = float(bw) / float(bh)
            if 0.25 <= aspect_ratio <= 3.8:
                if area > best_motion_area:
                    best_motion_area = area
                    airborne_candidate = (bx, by, bw, bh)

    detected = False
    ai_verified = False
    detected_label = "Trash"
    current_z = None

    if airborne_candidate is not None:
        bx, by, bw, bh = airborne_candidate

        pad = 20
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(FRAME_W, bx + bw + pad)
        y2 = min(FRAME_H, by + bh + pad)

        roi_patch = frame[y1:y2, x1:x2]

        if roi_patch.shape[0] > 10 and roi_patch.shape[1] > 10:
            ai_results = model.predict(
                source=roi_patch,
                conf=AI_CONFIDENCE,
                verbose=False,
                imgsz=160
            )

            for r in ai_results:
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    if len(TRASH_CLASSES) == 0 or cls_id in TRASH_CLASSES:
                        ai_verified = True
                        detected_label = f"{model.names[cls_id]} {float(b.conf[0]):.2f}"
                        break
                    elif cls_id in [32, 47, 49, 41]:
                        ai_verified = True
                        detected_label = f"{model.names[cls_id]} {float(b.conf[0]):.2f}"
                        break

        if not ai_verified and best_motion_area > 200:
            ai_verified = True
            detected_label = "Motion-Fast"

        if ai_verified:
            u = bx + bw // 2
            v = by + bh // 2
            pixel_diam = min(bw, bh)

            # Upward axis: Z is vertical height above camera lens
            Z_real = (FOCAL_LENGTH * TARGET_REAL_DIAMETER) / max(pixel_diam, 1)

            if MIN_PROXIMITY_Z <= Z_real <= MAX_PROXIMITY_Z:
                if now - last_detect_time > 0.60:
                    times, pts_x, pts_y, pts_z = [], [], [], []
                    predicted_impact = None

                valid_point = True
                if len(pts_z) > 0:
                    dt = now - times[-1]
                    if dt > 0.001:
                        dz_dt = abs(Z_real - pts_z[-1]) / dt
                        if dz_dt > MAX_DELTA_Z_SPEED:
                            valid_point = False

                if valid_point:
                    # Planar lateral coordinates relative to optical center
                    X_real = ((u - cx_img) * Z_real) / FOCAL_LENGTH
                    Y_real = -((v - cy_img) * Z_real) / FOCAL_LENGTH

                    times.append(now)
                    pts_x.append(X_real)
                    pts_y.append(Y_real)
                    pts_z.append(Z_real)
                    last_detect_time = now
                    detected = True
                    current_z = Z_real

                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.circle(frame, (u, v), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"[{detected_label}] H:{Z_real:.2f}m",
                                (bx, max(20, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                    if Z_real > PREDICTION_FREEZE_Z and len(times) >= 3:
                        res = solve_3d_trajectory_upward(times, pts_x, pts_y, pts_z, RIM_PLANE_Z)
                        if res:
                            predicted_impact = res

    if not detected and (now - last_detect_time > 0.45):
        predicted_impact = None

    # ----------------- HUD & SIMULATION -----------------
    target_x_rad, target_y_rad = None, None

    if predicted_impact is not None:
        t_x, t_y, t_rem = predicted_impact
        target_x_rad, target_y_rad = t_x, t_y

        # Screen projection scales dynamically using current tracked height
        ref_z = current_z if current_z is not None else 0.50
        target_px_u = int(cx_img + (t_x * FOCAL_LENGTH) / max(ref_z, 0.2))
        target_px_v = int(cy_img - (t_y * FOCAL_LENGTH) / max(ref_z, 0.2))

        target_px_u = np.clip(target_px_u, 20, FRAME_W - 20)
        target_px_v = np.clip(target_px_v, 20, FRAME_H - 20)

        cv2.putText(frame, "VERTICAL TRAJECTORY SOLVED!", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(frame, f"Offset X: {t_x:+.2f} m", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(frame, f"Offset Y: {t_y:+.2f} m", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
        cv2.putText(frame, f"Time to Rim: {t_rem*1000:.0f} ms", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1)

        cv2.drawMarker(frame, (target_px_u, target_px_v), (0, 0, 255),
                       cv2.MARKER_CROSS, 24, 2)
        cv2.circle(frame, (target_px_u, target_px_v), 18, (0, 255, 255), 1)

    draw_upward_radar(frame, target_x_rad, target_y_rad)

    loop_fps = 1.0 / max(time.time() - t_start, 0.001)
    cv2.putText(frame, f"FPS: {loop_fps:.0f}", (20, FRAME_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Upward-Facing 3D Trajectory Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()