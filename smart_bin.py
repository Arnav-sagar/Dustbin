import socket
import time
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from ultralytics import YOLO

# ----------------- CALIBRATION & HARDWARE -----------------
CAMERA_SOURCE = 1            # Auto-falls back across indices 0, 1, 2 if needed
FRAME_W = 640
FRAME_H = 480

# Standardized projectile diameter (e.g., 7.0 cm neon-taped ball / 5.0 cm ball)
TARGET_REAL_DIAMETER = 0.050
FOCAL_LENGTH = 415.0

# Wall-Mount Physical Geometry
WALL_CAMERA_HEIGHT = 1.90      # Height of camera lens from floor (meters)
CAMERA_PITCH_DEG = 35.0        # Downward tilt angle in degrees
BIN_RIM_HEIGHT = 0.45          # Catch rim height from floor (meters)
SUMMON_HAND_HEIGHT = 1.35      # Assumed height of raised palm from floor (meters)
SUMMON_STANDOFF_DIST = 0.45    # Stop 45 cm in front of person's feet

# Crowd & Motion Gates
HORIZON_CROWD_CUTOFF = int(FRAME_H * 0.20)
MIN_AIRBORNE_AREA = 80
MAX_HUMAN_BLOB_AREA = 3200
BORDER_MARGIN = 8

# AI & Gesture Settings
TRASH_CLASSES = [32, 39, 41]   # 32: sports ball, 39: bottle, 41: cup
AI_CONFIDENCE = 0.15
SUMMON_TRIGGER_FRAMES = 25     # ~0.8 seconds hold for palm summon

# ESP32 Wi-Fi UDP Bridge (Set ENABLE_UDP = True when ESP32 is connected)
ENABLE_UDP = False
ESP32_IP = "192.168.4.1"       # Default ESP32 SoftAP IP (or router-assigned IP)
ESP32_PORT = 4210
# ----------------------------------------------------------

theta = np.radians(CAMERA_PITCH_DEG)
cos_t = np.cos(theta)
sin_t = np.sin(theta)

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if ENABLE_UDP else None

def send_to_esp32(mode, target_xw, target_zw):
    """Streams target floor coordinates (meters) to ESP32 over UDP."""
    if ENABLE_UDP and udp_sock is not None:
        # Format: "CATCH,+0.150,1.850" or "SUMMON,-0.400,1.600"
        msg = f"{mode},{target_xw:+.3f},{target_zw:.3f}".encode("utf-8")
        try:
            udp_sock.sendto(msg, (ESP32_IP, ESP32_PORT))
        except Exception:
            pass

def open_working_camera(preferred_index=0):
    """Finds and opens the first working camera on Windows (DSHOW or MSMF)."""
    indices_to_try = [preferred_index] + [i for i in range(4) if i != preferred_index]
    backends = [
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("DEFAULT", cv2.CAP_ANY),
    ]

    for idx in indices_to_try:
        for backend_name, backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    print(f"[CAMERA] Connected on index {idx} using {backend_name}")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Attempt manual exposure lock (ignored safely if unsupported)
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
                    return cap
            cap.release()

    raise RuntimeError(
        "Could not open any webcam! Close the Windows Camera app, DroidCam preview, "
        "or browser tabs using the camera and run again."
    )

# Load YOLOv8 Nano
print("Loading YOLOv8 Nano...")
yolo_model = YOLO("yolov8n.pt")

# Load MediaPipe HandLandmarker from memory buffer (avoids OneDrive/path locks)
MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"hand_landmarker.task not found at {MODEL_PATH}. "
        "Download it into your script directory first."
    )

with open(MODEL_PATH, "rb") as f:
    hand_model_bytes = f.read()

base_options = python.BaseOptions(model_asset_buffer=hand_model_bytes)
hand_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# Open Camera & Create Visible Window
cap = open_working_camera(CAMERA_SOURCE)
WINDOW_NAME = "Smart Bin Dual-Mode Tracker"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=20, varThreshold=16, detectShadows=False
)

# ----------------- COORDINATE CONVERSIONS -----------------
def camera_to_world(xc, yc, zc):
    """Rotates tilted camera optical coordinates into room coordinates."""
    xw = xc
    yw = (cos_t * yc) - (sin_t * zc) + WALL_CAMERA_HEIGHT
    zw = (sin_t * yc) + (cos_t * zc)
    return np.array([[xw], [yw], [zw]])

def world_to_image_pixel(xw, yw, zw):
    """Projects 3D room coordinates back onto the 2D camera viewport."""
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

def ray_to_floor_summon(u, v):
    """Intersects the camera ray through wrist pixel (u, v) with hand plane to find floor spot."""
    cx_img, cy_img = FRAME_W // 2, FRAME_H // 2
    x_norm = (u - cx_img) / FOCAL_LENGTH
    y_norm = -(v - cy_img) / FOCAL_LENGTH

    dy = (y_norm * cos_t) - sin_t
    dz = (y_norm * sin_t) + cos_t

    if abs(dy) < 1e-4:
        return None

    t_ray = (SUMMON_HAND_HEIGHT - WALL_CAMERA_HEIGHT) / dy
    if t_ray <= 0.2 or t_ray > 6.0:
        return None

    person_xw = t_ray * x_norm
    person_zw = (t_ray * dz) - SUMMON_STANDOFF_DIST

    if person_zw < 0.3 or person_zw > 4.5 or abs(person_xw) > 2.2:
        return None

    return float(person_xw), float(person_zw)

# ----------------- 3D BALLISTIC KALMAN ESTIMATOR -----------------
class WallKalmanTracker:
    def __init__(self):
        self.initialized = False
        self.last_t = None
        # State: [Xw, Yw, Zw, Vx, Vy, Vz]^T
        self.state = np.zeros((6, 1))
        self.P = np.eye(6) * 1.0
        self.Q = np.diag([0.02, 0.02, 0.02, 0.4, 0.2, 0.4])
        # Higher R on Yw/Zw suppresses bounding-box tumble jitter
        self.R = np.diag([0.005, 0.06, 0.08])
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

        if landing_zw < 0.2 or landing_zw > 5.0 or abs(landing_xw) > 2.5:
            return None

        return float(landing_xw), float(landing_zw), float(t_impact)

def draw_floor_target_box(frame, target_xw, target_zw, plane_yw, color_bgr, label_text):
    """Draws a perspective-aligned 35x35 cm target square on the floor/rim plane."""
    half = 0.175
    corners_3d = [
        (target_xw - half, plane_yw, target_zw - half),
        (target_xw + half, plane_yw, target_zw - half),
        (target_xw + half, plane_yw, target_zw + half),
        (target_xw - half, plane_yw, target_zw + half),
    ]

    pts_2d = []
    for (cx_w, cy_w, cz_w) in corners_3d:
        proj = world_to_image_pixel(cx_w, cy_w, cz_w)
        if proj is None:
            return
        pts_2d.append([proj[0], proj[1]])

    poly_pts = np.array(pts_2d, np.int32).reshape((-1, 1, 2))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly_pts], color_bgr)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.polylines(frame, [poly_pts], isClosed=True, color=color_bgr, thickness=2)

    center_proj = world_to_image_pixel(target_xw, plane_yw, target_zw)
    if center_proj is not None:
        cu, cv_pt, _ = center_proj
        cv2.drawMarker(frame, (cu, cv_pt), color_bgr, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(frame, label_text, (cu - 45, cv_pt - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 2)

# ----------------- STATE VARIABLES -----------------
tracker = WallKalmanTracker()
update_count = 0
last_detect_time = 0
trajectory_locked = False
locked_landing = None

palm_hold_frames = 0
summon_target = None
summon_clear_time = 0
frame_counter = 0

print("Starting tracking loop. Press 'q' in the video window to quit.")

while True:
    t_start = time.time()
    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.01)
        continue

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    cx_img, cy_img = FRAME_W // 2, FRAME_H // 2
    now = time.time()
    frame_counter += 1

    # ----------------- 1. BALLISTIC TOSS DETECTION (PRIORITY) -----------------
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

            if bx <= BORDER_MARGIN or by_full <= (HORIZON_CROWD_CUTOFF + BORDER_MARGIN) or \
               (bx + bw) >= (FRAME_W - BORDER_MARGIN) or (by_full + bh) >= (FRAME_H - BORDER_MARGIN):
                continue

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
        x1, y1 = max(0, bx - pad), max(0, by_f - pad)
        x2, y2 = min(FRAME_W, bx + bw + pad), min(FRAME_H, by_f + bh + pad)
        roi_patch = frame[y1:y2, x1:x2]

        ai_verified = False
        is_human = False

        if roi_patch.shape[0] > 10 and roi_patch.shape[1] > 10:
            ai_results = yolo_model.predict(source=roi_patch, conf=AI_CONFIDENCE, verbose=False, imgsz=160)
            for r in ai_results:
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    if cls_id == 0:
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
                    trajectory_locked = False
                    locked_landing = None

                summon_target = None
                palm_hold_frames = 0

                X_cam = ((u - cx_img) * Z_cam) / FOCAL_LENGTH
                Y_cam = -((v - cy_img) * Z_cam) / FOCAL_LENGTH
                pos_world = camera_to_world(X_cam, Y_cam, Z_cam)

                tracker.update(pos_world, now)
                update_count += 1
                last_detect_time = now
                detected = True

                cv2.rectangle(frame, (bx, by_f), (bx + bw, by_f + bh), (255, 255, 0), 2)
                cv2.circle(frame, (int(u), int(v)), 4, (0, 0, 255), -1)

                if update_count >= 5 and not trajectory_locked:
                    res = tracker.predict_landing(BIN_RIM_HEIGHT)
                    if res and res[2] > 0.12:
                        locked_landing = res
                        trajectory_locked = True
                        send_to_esp32("CATCH", res[0], res[1])

    if not detected and (now - last_detect_time > 0.50):
        trajectory_locked = False
        locked_landing = None

    # ----------------- 2. GESTURE SUMMON MODE (IDLE ONLY) -----------------
    if not detected and not trajectory_locked and (frame_counter % 2 == 0):
        rgb_square = np.ascontiguousarray(
            cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2RGB)
        )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_square)
        hand_res = hand_detector.detect(mp_image)

        if hand_res.hand_landmarks:
            lm = hand_res.hand_landmarks[0]
            fingers_open = (
                lm[8].y < lm[6].y
                and lm[12].y < lm[10].y
                and lm[16].y < lm[14].y
                and lm[20].y < lm[18].y
            )
            if fingers_open:
                palm_hold_frames += 2
                wrist_u = int(lm[0].x * FRAME_W)
                wrist_v = int(lm[0].y * FRAME_H)

                progress = min(1.0, palm_hold_frames / SUMMON_TRIGGER_FRAMES)
                cv2.ellipse(frame, (wrist_u, wrist_v), (22, 22), 0, 0, int(360 * progress), (255, 180, 0), 3)

                if palm_hold_frames >= SUMMON_TRIGGER_FRAMES:
                    floor_pt = ray_to_floor_summon(wrist_u, wrist_v)
                    if floor_pt is not None:
                        summon_target = floor_pt
                        summon_clear_time = now + 4.0
                        send_to_esp32("SUMMON", floor_pt[0], floor_pt[1])
                        palm_hold_frames = 0
            else:
                palm_hold_frames = max(0, palm_hold_frames - 2)
        else:
            palm_hold_frames = max(0, palm_hold_frames - 1)

    if summon_target is not None and now > summon_clear_time:
        summon_target = None

    # ----------------- 3. HUD & FLOOR TARGET RENDERING -----------------
    cv2.line(frame, (0, HORIZON_CROWD_CUTOFF), (FRAME_W, HORIZON_CROWD_CUTOFF), (0, 0, 255), 1)

    if trajectory_locked and locked_landing is not None:
        txw, tzw, trem = locked_landing
        draw_floor_target_box(frame, txw, tzw, BIN_RIM_HEIGHT, (0, 255, 0), f"CATCH ({trem*1000:.0f}ms)")
        cv2.putText(frame, f"CATCH LOCKED -> Xw:{txw:+.2f}m  Zw:{tzw:.2f}m", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)

    elif summon_target is not None:
        sxw, szw = summon_target
        draw_floor_target_box(frame, sxw, szw, 0.0, (255, 180, 0), "SUMMON BIN HERE")
        cv2.putText(frame, f"GESTURE SUMMON -> Xw:{sxw:+.2f}m  Zw:{szw:.2f}m", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 180, 0), 2)

    else:
        cv2.putText(frame, "READY: Toss Object or Hold Open Palm to Summon", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    loop_fps = 1.0 / max(time.time() - t_start, 0.001)
    cv2.putText(frame, f"FPS: {loop_fps:.0f}", (20, FRAME_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()