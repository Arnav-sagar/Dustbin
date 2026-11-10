import time
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------- CALIBRATION & HARDWARE -----------------
CAMERA_SOURCE = 1              # 0 or 1 for USB / DroidCam / Iriun
FRAME_W = 640
FRAME_H = 480

TARGET_REAL_DIAMETER = 0.050   # 5.0 cm real physical diameter (paper wad / ball)
FOCAL_LENGTH = 512.2           # Calibrated focal length

# Physical Wall Measurements
WALL_CAMERA_HEIGHT = 1.40      # Height of camera lens from the floor (meters)
CAMERA_PITCH_DEG = 35.0        # Downward tilt angle in degrees
BIN_RIM_HEIGHT = 0.45          # Height of the target landing / bin rim (meters)

# Spatial & Crowd Gates
HORIZON_CROWD_CUTOFF = int(FRAME_H * 0.20)  # Top 20% ignored to remove background crowds
MIN_AIRBORNE_AREA = 80
MAX_HUMAN_BLOB_AREA = 3200
BORDER_MARGIN = 8

TRASH_CLASSES = [32, 39, 41]   # 32: sports ball, 39: bottle, 41: cup
AI_CONFIDENCE = 0.15
# ----------------------------------------------------------

theta = np.radians(CAMERA_PITCH_DEG)
cos_t = np.cos(theta)
sin_t = np.sin(theta)

print("Loading YOLOv8 Nano...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Disable auto-exposure flickering
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=20, varThreshold=16, detectShadows=False
)

# ----------------- COORDINATE CONVERSIONS -----------------
def camera_to_world(xc, yc, zc):
    """Rotates tilted camera optical vectors into true room coordinates."""
    xw = xc
    yw = (cos_t * yc) - (sin_t * zc) + WALL_CAMERA_HEIGHT
    zw = (sin_t * yc) + (cos_t * zc)
    return np.array([[xw], [yw], [zw]])

def world_to_image_pixel(xw, yw, zw):
    """Projects 3D room coordinates back onto the 2D camera viewport."""
    # Inverse pitch transform
    y_shifted = yw - WALL_CAMERA_HEIGHT
    xc = xw
    yc = (cos_t * y_shifted) + (sin_t * zw)
    zc = (-sin_t * y_shifted) + (cos_t * zw)

    if zc <= 0.1:
        return None

    cx_img, cy_img = FRAME_W // 2, FRAME_H // 2
    u = int(cx_img + (xc * FOCAL_LENGTH) / zc)
    v = int(cy_img - (yc * FOCAL_LENGTH) / zc)
    return u, v, zc

# ----------------- 3D BALLISTIC KALMAN ESTIMATOR -----------------
class WallKalmanTracker:
    def __init__(self):
        self.initialized = False
        self.last_t = None
        # State: [Xw, Yw, Zw, Vx, Vy, Vz]^T
        self.state = np.zeros((6, 1))
        self.P = np.eye(6) * 1.0
        self.Q = np.diag([0.02, 0.02, 0.02, 0.4, 0.2, 0.4])
        self.R = np.diag([0.005, 0.02, 0.02])
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

    def reset(self):
        self.initialized = False
        self.last_t = None
        self.state = np.zeros((6, 1))
        self.P = np.eye(6) * 1.0

    def update(self, pos_world, t_now):
        if not self.initialized:
            self.state[0:3] = pos_world
            self.state[3:6] = 0.0
            self.last_t = t_now
            self.initialized = True
            return

        dt = t_now - self.last_t
        if dt <= 0.0001:
            return
        self.last_t = t_now

        g = 9.81
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Apply gravity strictly along Yw (vertical floor axis)
        self.state = F @ self.state
        self.state[1, 0] -= 0.5 * g * (dt ** 2)
        self.state[4, 0] -= g * dt

        self.P = F @ self.P @ F.T + self.Q

        y_res = pos_world - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + (K @ y_res)
        self.P = (np.eye(6) - (K @ self.H)) @ self.P

    def predict_landing(self, target_plane_yw):
        if not self.initialized:
            return None

        xw, yw, zw = self.state[0, 0], self.state[1, 0], self.state[2, 0]
        vx, vy, vz = self.state[3, 0], self.state[4, 0], self.state[5, 0]

        g = 9.81
        # Solve: -0.5*g*t^2 + vy*t + (yw - target_plane_yw) = 0
        A = -0.5 * g
        B = vy
        C = yw - target_plane_yw

        disc = B**2 - 4 * A * C
        if disc <= 0:
            return None

        t_impact = (-B - np.sqrt(disc)) / (2.0 * A)
        if t_impact <= 0.02 or t_impact > 2.5:
            return None

        landing_xw = xw + vx * t_impact
        landing_zw = zw + vz * t_impact

        # Bound check: Landing must be within realistic room area
        if landing_zw < 0.2 or landing_zw > 5.0 or abs(landing_xw) > 2.5:
            return None

        return float(landing_xw), float(landing_zw), float(t_impact)

tracker = WallKalmanTracker()
update_count = 0
last_detect_time = 0
predicted_landing = None

# Flush initial camera buffer
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

    # Crop crowd background above horizon
    active_slice = frame[HORIZON_CROWD_CUTOFF:, :]

    fg_mask = bg_subtractor.apply(active_slice)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AIRBORNE_AREA < area < MAX_HUMAN_BLOB_AREA:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            by_full = by + HORIZON_CROWD_CUTOFF

            # Reject boundary-attached arms/legs
            if bx <= BORDER_MARGIN or by_full <= (HORIZON_CROWD_CUTOFF + BORDER_MARGIN) or \
               (bx + bw) >= (FRAME_W - BORDER_MARGIN) or (by_full + bh) >= (FRAME_H - BORDER_MARGIN):
                continue

            # Compactness check (reject irregular human torsos)
            hull = cv2.convexHull(cnt)
            solidity = float(area) / max(cv2.contourArea(hull), 1)
            if solidity < 0.75:
                continue

            if area > best_area:
                best_area = area
                candidate = (bx, by_full, bw, bh, cnt)

    detected = False

    if candidate is not None:
        bx, by_f, bw, bh, cnt = candidate

        pad = 20
        x1 = max(0, bx - pad)
        y1 = max(0, by_f - pad)
        x2 = min(FRAME_W, bx + bw + pad)
        y2 = min(FRAME_H, by_f + bh + pad)

        roi_patch = frame[y1:y2, x1:x2]
        ai_verified = False
        is_human = False

        if roi_patch.shape[0] > 10 and roi_patch.shape[1] > 10:
            ai_results = model.predict(source=roi_patch, conf=AI_CONFIDENCE, verbose=False, imgsz=160)
            for r in ai_results:
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    if cls_id == 0:  # Class 0 = Person veto
                        is_human = True
                        break
                    elif len(TRASH_CLASSES) == 0 or cls_id in TRASH_CLASSES or cls_id in [32, 47, 49, 41]:
                        ai_verified = True
                        break

        if not is_human and not ai_verified and best_area > 180:
            ai_verified = True

        if ai_verified and not is_human:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                u = M["m10"] / M["m00"]
                v = M["m01"] / M["m00"] + HORIZON_CROWD_CUTOFF
            else:
                u = bx + bw / 2.0
                v = by_f + bh / 2.0

            pixel_diam = 2.0 * np.sqrt(best_area / np.pi)
            Z_cam = (FOCAL_LENGTH * TARGET_REAL_DIAMETER) / max(pixel_diam, 1.0)

            if 0.35 <= Z_cam <= 5.0:
                if now - last_detect_time > 0.60:
                    tracker.reset()
                    update_count = 0
                    predicted_landing = None

                X_cam = ((u - cx_img) * Z_cam) / FOCAL_LENGTH
                Y_cam = -((v - cy_img) * Z_cam) / FOCAL_LENGTH

                pos_world = camera_to_world(X_cam, Y_cam, Z_cam)

                tracker.update(pos_world, now)
                update_count += 1
                last_detect_time = now
                detected = True

                # Draw current object tracking box (Cyan)
                cv2.rectangle(frame, (bx, by_f), (bx + bw, by_f + bh), (255, 255, 0), 2)
                cv2.circle(frame, (int(u), int(v)), 4, (0, 0, 255), -1)

                if update_count >= 4:
                    res = tracker.predict_landing(BIN_RIM_HEIGHT)
                    if res:
                        predicted_landing = res

    if not detected and (now - last_detect_time > 0.45):
        predicted_landing = None

    # Crowd Gate Horizon line
    cv2.line(frame, (0, HORIZON_CROWD_CUTOFF), (FRAME_W, HORIZON_CROWD_CUTOFF), (0, 0, 255), 1)

    # ----------------- PROJECTED GREEN LANDING BOX -----------------
    if predicted_landing is not None:
        target_xw, target_zw, t_rem = predicted_landing

        # Construct a 35 cm x 35 cm landing target on the catch plane
        box_half_size = 0.175  # meters
        corners_3d = [
            (target_xw - box_half_size, BIN_RIM_HEIGHT, target_zw - box_half_size),
            (target_xw + box_half_size, BIN_RIM_HEIGHT, target_zw - box_half_size),
            (target_xw + box_half_size, BIN_RIM_HEIGHT, target_zw + box_half_size),
            (target_xw - box_half_size, BIN_RIM_HEIGHT, target_zw + box_half_size)
        ]

        # Project 3D floor square into 2D camera viewport
        pts_2d = []
        valid_projection = True
        for (cx_w, cy_w, cz_w) in corners_3d:
            proj = world_to_image_pixel(cx_w, cy_w, cz_w)
            if proj is None:
                valid_projection = False
                break
            pts_2d.append([proj[0], proj[1]])

        if valid_projection:
            poly_pts = np.array(pts_2d, np.int32).reshape((-1, 1, 2))

            # Semi-transparent green landing zone overlay
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly_pts], (0, 255, 0))
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

            # Solid green perimeter border
            cv2.polylines(frame, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Center target crosshair
            center_proj = world_to_image_pixel(target_xw, BIN_RIM_HEIGHT, target_zw)
            if center_proj is not None:
                cu, cv_pt, _ = center_proj
                cv2.drawMarker(frame, (cu, cv_pt), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
                cv2.putText(frame, f"FALL: {t_rem*1000:.0f}ms", (cu - 35, cv_pt - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # On-screen HUD coordinates
        cv2.putText(frame, "LANDING ESTIMATED", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"Floor Lateral (Xw): {target_xw:+.2f} m", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(frame, f"Floor Depth   (Zw): {target_zw:.2f} m", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)

    loop_fps = 1.0 / max(time.time() - t_start, 0.001)
    cv2.putText(frame, f"FPS: {loop_fps:.0f}", (20, FRAME_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Wall-Mounted Projectile Landing Predictor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()