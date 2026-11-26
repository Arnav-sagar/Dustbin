import socket
import time
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# ================= CAMERA & GEOMETRY CALIBRATION =================
CAMERA_SOURCE = 1              # Auto-falls back across indices 0..3
FRAME_W = 640
FRAME_H = 480

# Calibrated for 22.9mm equiv / 85 deg FOV camera at 640x480
FOCAL_LENGTH = 415.0
TARGET_REAL_DIAMETER = 0.065   # ~6.5 cm crumpled A4 fluorescent paper ball

# Set to match WHERE THE CAMERA ACTUALLY IS right now:
# - On desk testing: WALL_CAMERA_HEIGHT = 0.85, CAMERA_PITCH_DEG = -10.0 (tilted up)
# - On exhibition wall: WALL_CAMERA_HEIGHT = 1.90, CAMERA_PITCH_DEG = 35.0 (tilted down)
# You can tune both live: '[' / ']' for Pitch, 'w' / 's' for Height!
WALL_CAMERA_HEIGHT = 0.85      # Height of camera lens from floor (meters)
CAMERA_PITCH_DEG = -10.0       # Downward tilt (+deg = down, -deg = tilted up)
BIN_RIM_HEIGHT = 0.45          # Catch rim height from floor (meters)
SUMMON_HAND_HEIGHT = 1.35      # Assumed height of raised palm from floor (meters)
SUMMON_STANDOFF_DIST = 0.45    # Stop 45 cm in front of person's feet
MIN_RELEASE_HEIGHT = 0.55      # Ignores stationary balls low near floor/basket

# Fluorescent Paper Color Mode ("PINK" recommended, or "YELLOW")
# STRICT blocks clutter; BLUR only activates on actively moving pixels
PAPER_COLOR = "PINK"
HSV_STRICT = {
    "PINK":   (np.array([140, 115, 110]), np.array([175, 255, 255])),
    "YELLOW": (np.array([20,  125, 120]), np.array([35,  255, 255])),
}
HSV_BLUR = {
    "PINK":   (np.array([135, 45,  80]),  np.array([179, 255, 255])),
    "YELLOW": (np.array([16,  55,  90]),  np.array([40,  255, 255])),
}
STRICT_LOW, STRICT_HIGH = HSV_STRICT[PAPER_COLOR]
BLUR_LOW, BLUR_HIGH = HSV_BLUR[PAPER_COLOR]

# Detection & Motion Gates (HORIZON_CROWD_CUTOFF = 0 so high drops aren't clipped!)
HORIZON_CROWD_CUTOFF = 0
MIN_PAPER_AREA = 85
MAX_PAPER_AREA = 5000
MOTION_DIFF_THRESH = 18
BORDER_MARGIN = 4

# Gesture Summon Settings
SUMMON_TRIGGER_FRAMES = 25

# ESP32 / Wemos D1 R1 Wi-Fi UDP Bridge (Set ENABLE_UDP = True when ready)
ENABLE_UDP = False
ESP_IP = "192.168.4.1"
ESP_PORT = 4210
# =================================================================

theta = np.radians(CAMERA_PITCH_DEG)
cos_t = np.cos(theta)
sin_t = np.sin(theta)

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if ENABLE_UDP else None


def update_pitch(new_pitch_deg):
    global CAMERA_PITCH_DEG, theta, cos_t, sin_t
    CAMERA_PITCH_DEG = float(np.clip(new_pitch_deg, -35.0, 75.0))
    theta = np.radians(CAMERA_PITCH_DEG)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)


def update_height(new_height_m):
    global WALL_CAMERA_HEIGHT
    WALL_CAMERA_HEIGHT = float(np.clip(new_height_m, 0.20, 3.50))


def send_to_esp(mode, target_xw, target_zw):
    if ENABLE_UDP and udp_sock is not None:
        msg = f"{mode},{target_xw:+.3f},{target_zw:.3f}".encode("utf-8")
        try:
            udp_sock.sendto(msg, (ESP_IP, ESP_PORT))
        except Exception:
            pass


def open_working_camera(preferred_index=0):
    indices_to_try = [preferred_index] + [i for i in range(4) if i != preferred_index]
    for idx in indices_to_try:
        for backend_name, backend in [("CAP_DSHOW", cv2.CAP_DSHOW), ("DEFAULT", cv2.CAP_ANY)]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    print(f"[CAMERA] Connected on index {idx} using {backend_name}")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap
            cap.release()
    raise RuntimeError("No working camera found! Close other apps using the camera and retry.")


def find_hand_model():
    candidates = [
        Path(__file__).resolve().parent / "hand_landmarker.task",
        Path(r"C:\Users\ACER\OneDrive\Desktop\Dustbin\Dustbin\hand_landmarker.task"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("hand_landmarker.task not found!")


# ----------------- COORDINATE CONVERSIONS -----------------
def camera_to_world(xc, yc, zc):
    """Rotates camera coordinates into room space (Yw=0 is floor)."""
    xw = xc
    yw = (cos_t * yc) - (sin_t * zc) + WALL_CAMERA_HEIGHT
    zw = (sin_t * yc) + (cos_t * zc)
    return float(xw), float(yw), float(zw)


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
    return u, v, float(zc)


def ray_to_floor_summon(u, v):
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


# ----------------- BALLISTIC PARABOLA FITTER -----------------
class BallisticParabolaTracker:
    def __init__(self):
        self.samples = []  # (t, xw, yw, zw)

    def reset(self):
        self.samples.clear()

    def count(self):
        return len(self.samples)

    def update(self, pos_world, t_now):
        xw, yw, zw = pos_world
        if self.samples:
            last_t, lx, ly, lz = self.samples[-1]
            dt = t_now - last_t
            if dt <= 0.002:
                return False
            dist = np.sqrt((xw - lx) ** 2 + (yw - ly) ** 2 + (zw - lz) ** 2)
            if (dist / dt) > 16.0:
                return False

        self.samples.append((t_now, xw, yw, zw))
        if len(self.samples) > 8:
            self.samples.pop(0)
        return True

    def predict_landing(self, target_plane_yw):
        if len(self.samples) < 4:
            return None

        t0 = self.samples[0][0]
        t_arr = np.array([s[0] - t0 for s in self.samples], dtype=np.float64)
        if t_arr[-1] < 0.04:
            return None

        x_arr = np.array([s[1] for s in self.samples], dtype=np.float64)
        y_arr = np.array([s[2] for s in self.samples], dtype=np.float64)
        z_arr = np.array([s[3] for s in self.samples], dtype=np.float64)

        g = 9.81
        y_linear = y_arr + 0.5 * g * (t_arr ** 2)

        vx, x0 = np.polyfit(t_arr, x_arr, 1)
        vy, y0 = np.polyfit(t_arr, y_linear, 1)
        vz, z0 = np.polyfit(t_arr, z_arr, 1)

        # Reject random jitter & slow hand lifting (tighter parabolic check)
        pred_x = x0 + vx * t_arr
        pred_y = y0 + vy * t_arr - 0.5 * g * (t_arr ** 2)
        rms_err = np.sqrt(np.mean((x_arr - pred_x) ** 2 + (y_arr - pred_y) ** 2))
        if rms_err > 0.14:
            return None

        # Current vertical velocity at latest frame
        vy_now = vy - g * t_arr[-1]
        speed_3d = np.sqrt(vx ** 2 + vy_now ** 2 + vz ** 2)

        # Require real throw speed (or clear free-fall downward motion)
        if speed_3d < 1.15:
            return None

        # Ignore slowly raising your hand upward while holding the ball
        if vy_now > 0.2 and abs(vx) < 0.7 and abs(vz) < 0.7:
            return None

        A = -0.5 * g
        B = vy
        C = y0 - target_plane_yw
        disc = B ** 2 - 4.0 * A * C
        if disc <= 0:
            return None

        t_hit = (-B - np.sqrt(disc)) / (2.0 * A)
        t_rem = t_hit - t_arr[-1]
        if t_rem <= 0.05 or t_rem > 2.2:
            return None

        landing_xw = x0 + vx * t_hit
        landing_zw = z0 + vz * t_hit

        if landing_zw < 0.2 or landing_zw > 5.0 or abs(landing_xw) > 2.5:
            return None

        return float(landing_xw), float(landing_zw), float(t_rem)


def draw_floor_target_box(frame, target_xw, target_zw, plane_yw, color_bgr, label_text):
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


# ----------------- INITIALIZE MODELS & CAMERA -----------------
MODEL_PATH = find_hand_model()
with open(MODEL_PATH, "rb") as f:
    hand_model_bytes = f.read()

hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=hand_model_bytes),
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
)

cap = open_working_camera(CAMERA_SOURCE)
WINDOW_NAME = "Smart Bin Wall-Mount Tracker"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)

KERNEL_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
KERNEL_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ----------------- STATE VARIABLES -----------------
tracker = BallisticParabolaTracker()
last_detect_time = 0.0
trajectory_locked = False
locked_landing = None
lock_expire_time = 0.0

palm_hold_frames = 0
summon_target = None
summon_clear_time = 0.0
frame_counter = 0
show_mask_debug = False
prev_gray = None

print(f"Blur-Immune Mode | Color: {PAPER_COLOR} | f={FOCAL_LENGTH}px")
print("Keys: 'm'=Mask | '['/']'=Pitch | 'w'/'s'=Cam Height | 'q'=Quit")

while True:
    t_start = time.time()
    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.005)
        continue

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    cx_img, cy_img = FRAME_W // 2, FRAME_H // 2
    now = time.time()
    frame_counter += 1

    # Auto-clear expired catch lock (prevents stuck box when picking up the ball!)
    if trajectory_locked and now > lock_expire_time:
        trajectory_locked = False
        locked_landing = None
        tracker.reset()

    # ----------------- 1. MOTION-GATED BLUR-IMMUNE PAPER DETECTION -----------------
    active_slice = frame[HORIZON_CROWD_CUTOFF:, :]
    curr_gray = cv2.GaussianBlur(cv2.cvtColor(active_slice, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    if prev_gray is None:
        motion_mask = np.zeros_like(curr_gray)
    else:
        frame_diff = cv2.absdiff(curr_gray, prev_gray)
        _, motion_mask = cv2.threshold(frame_diff, MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(motion_mask, KERNEL_5, iterations=1)
    prev_gray = curr_gray

    hsv = cv2.cvtColor(active_slice, cv2.COLOR_BGR2HSV)

    # Strict mask uses 5x5 cleanup; motion-gated blur mask uses 3x3 to keep fast white-wall streaks
    strict_mask = cv2.inRange(hsv, STRICT_LOW, STRICT_HIGH)
    strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_OPEN, KERNEL_5)

    blur_mask = cv2.bitwise_and(cv2.inRange(hsv, BLUR_LOW, BLUR_HIGH), motion_mask)
    blur_mask = cv2.morphologyEx(blur_mask, cv2.MORPH_OPEN, KERNEL_3)

    color_mask = cv2.bitwise_or(strict_mask, blur_mask)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, KERNEL_5)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate = None
    best_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_PAPER_AREA < area < MAX_PAPER_AREA:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            by_full = by + HORIZON_CROWD_CUTOFF

            if bx <= BORDER_MARGIN or (bx + bw) >= (FRAME_W - BORDER_MARGIN):
                continue

            roi_motion = motion_mask[by:by + bh, bx:bx + bw]
            if cv2.countNonZero(roi_motion) < max(25, int(area * 0.20)):
                continue

            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            short_side = min(rw, rh)
            if short_side < 5.0:
                continue

            if area > best_area:
                best_area = area
                candidate = (bx, by_full, bw, bh, cnt, short_side)

    detected = False

    if candidate is not None:
        bx, by_f, bw, bh, cnt, pixel_diam = candidate
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            u = M["m10"] / M["m00"]
            v = (M["m01"] / M["m00"]) + HORIZON_CROWD_CUTOFF
        else:
            u = bx + bw / 2.0
            v = by_f + bh / 2.0

        Z_cam = (FOCAL_LENGTH * TARGET_REAL_DIAMETER) / max(pixel_diam, 5.0)

        if 0.35 <= Z_cam <= 5.0:
            X_cam = ((u - cx_img) * Z_cam) / FOCAL_LENGTH
            Y_cam = -((v - cy_img) * Z_cam) / FOCAL_LENGTH
            pos_world = camera_to_world(X_cam, Y_cam, Z_cam)

            if now - last_detect_time > 0.35:
                tracker.reset()

            if tracker.count() >= 2 or pos_world[1] >= MIN_RELEASE_HEIGHT:
                if tracker.update(pos_world, now):
                    summon_target = None
                    palm_hold_frames = 0
                    last_detect_time = now
                    detected = True

                    cv2.rectangle(frame, (bx, by_f), (bx + bw, by_f + bh), (255, 0, 255), 2)
                    cv2.circle(frame, (int(u), int(v)), 4, (0, 255, 255), -1)

                    if tracker.count() >= 4 and not trajectory_locked:
                        res = tracker.predict_landing(BIN_RIM_HEIGHT)
                        if res is not None:
                            locked_landing = res
                            trajectory_locked = True
                            # Auto-expire right after predicted impact + 0.55s display buffer
                            lock_expire_time = now + min(res[2] + 0.55, 1.8)
                            send_to_esp("CATCH", res[0], res[1])

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
                        send_to_esp("SUMMON", floor_pt[0], floor_pt[1])
                        palm_hold_frames = 0
            else:
                palm_hold_frames = max(0, palm_hold_frames - 2)
        else:
            palm_hold_frames = max(0, palm_hold_frames - 1)

    if summon_target is not None and now > summon_clear_time:
        summon_target = None

    # ----------------- 3. HUD & TARGET RENDERING -----------------
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
        cv2.putText(frame, f"[{PAPER_COLOR}] Toss Paper or Open Palm | '['/']':Pitch 'w'/'s':Height", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if show_mask_debug:
        mask_pip = cv2.cvtColor(cv2.resize(color_mask, (160, 120)), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mask_pip, (0, 0), (159, 119), (255, 0, 255), 1)
        frame[FRAME_H - 130:FRAME_H - 10, FRAME_W - 170:FRAME_W - 10] = mask_pip

    loop_fps = 1.0 / max(time.time() - t_start, 0.001)
    cv2.putText(frame, f"FPS:{loop_fps:.0f} | H:{WALL_CAMERA_HEIGHT:.2f}m P:{CAMERA_PITCH_DEG:+.0f}d",
                (20, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("m"):
        show_mask_debug = not show_mask_debug
    elif key == ord("["):
        update_pitch(CAMERA_PITCH_DEG - 1.0)
    elif key == ord("]"):
        update_pitch(CAMERA_PITCH_DEG + 1.0)
    elif key == ord("w"):
        update_height(WALL_CAMERA_HEIGHT + 0.05)
    elif key == ord("s"):
        update_height(WALL_CAMERA_HEIGHT - 0.05)

cap.release()
cv2.destroyAllWindows()