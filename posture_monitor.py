import os
import argparse
import platform
import subprocess
import math as m

import cv2
import numpy as np
import mediapipe as mp


ALERT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alert.png")
BAD_POSTURE_LIMIT_S = 20
WINDOW = 'Posture Monitor'
ALERT_WINDOW = 'Posture Alert'

# View detection: body rotation from the camera, in degrees (0 = facing it, 90 = side-on).
FRONT_MAX_YAW = 30        # below this, use the front-view method
SIDE_MIN_YAW = 60         # above this, use the side-view method; in between can't be judged
VIEW_SWITCH_S = 1.0       # new view must hold this long before switching (avoids flicker)

# Front-view thresholds, relative to the calibrated upright baseline.
CALIBRATION_S = 3
NECK_DROP = 0.80          # bad if ear-to-shoulder height falls below 80% of baseline
LEAN_IN = 1.15            # bad if shoulders look more than 15% wider (closer to camera)
SHOULDER_TILT_DEG = 10    # bad if shoulder line tilts more than this

# Side-view thresholds (degrees from vertical).
NECK_MAX_DEG = 12
TORSO_MAX_DEG = 10

font = cv2.FONT_HERSHEY_PLAIN
red = (50, 50, 255)
green = (127, 255, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)
white = (255, 255, 255)
orange = (0, 165, 255)

mp_pose = mp.solutions.pose
P = mp_pose.PoseLandmark
pose = mp_pose.Pose()


# Calculate distance
def findDistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Calculate angle between the segment (x1,y1)->(x2,y2) and the vertical.
def findAngle(x1, y1, x2, y2):
    cos_t = (y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1)
    theta = m.acos(max(-1.0, min(1.0, cos_t)))  # clamp rounding error
    return (180 / m.pi) * theta


ALERT_MSG = "You've had bad posture for %d seconds. Sit up straight!" % BAD_POSTURE_LIMIT_S


# Alert when bad posture is held too long. Uses a system-level popup so it appears
# on top of whatever app you're in; falls back to an OpenCV window if that fails.
def sendWarning():
    print("\a" + ALERT_MSG)
    system = platform.system()

    if system == "Darwin":
        script = [
            'tell application "System Events"',
            'activate',
            'beep',
            'display dialog "%s" with title "Posture Alert" buttons {"OK"} '
            'default button "OK" with icon caution' % ALERT_MSG,
            'end tell',
        ]
        args = ["osascript"]
        for line in script:
            args += ["-e", line]
        if subprocess.run(args, capture_output=True).returncode == 0:
            return

    elif system == "Windows":
        import ctypes
        MB_ICONWARNING, MB_SYSTEMMODAL, MB_TOPMOST, MB_SETFOREGROUND = 0x30, 0x1000, 0x40000, 0x10000
        ctypes.windll.user32.MessageBoxW(0, ALERT_MSG, "Posture Alert",
                                         MB_ICONWARNING | MB_SYSTEMMODAL | MB_TOPMOST | MB_SETFOREGROUND)
        return

    elif system == "Linux":
        # zenity gives a blocking dialog; notify-send gives a desktop notification.
        try:
            if subprocess.run(["zenity", "--warning", "--title=Posture Alert",
                               "--text=" + ALERT_MSG]).returncode in (0, 1):
                return
        except FileNotFoundError:
            pass
        try:
            subprocess.run(["notify-send", "-u", "critical", "Posture Alert", ALERT_MSG])
        except FileNotFoundError:
            pass

    show_alert_window()


# Fallback alert: OpenCV window. Closes on any key or the window's X.
def show_alert_window():
    img = cv2.imread(ALERT_IMAGE) if os.path.exists(ALERT_IMAGE) else None
    if img is None:
        img = np.full((360, 640, 3), (40, 40, 200), dtype=np.uint8)
        cv2.putText(img, 'FIX YOUR POSTURE', (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.6, white, 4)
        cv2.putText(img, 'Press any key to continue', (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2)

    cv2.namedWindow(ALERT_WINDOW, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(ALERT_WINDOW, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    cv2.imshow(ALERT_WINDOW, img)
    while True:
        if cv2.waitKey(100) != -1:
            break
        if cv2.getWindowProperty(ALERT_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow(ALERT_WINDOW)
    cv2.waitKey(1)


def visible(lm, points, min_vis=0.5):
    return all(lm.landmark[p].visibility > min_vis for p in points)


# ---------------------------------------- VIEW DETECTION ---------------------------------------- #
def body_yaw(world):
    """Rotation of the shoulder line away from the camera plane, 0 to 90 degrees."""
    ls, rs = world.landmark[P.LEFT_SHOULDER], world.landmark[P.RIGHT_SHOULDER]
    yaw = abs(m.degrees(m.atan2(rs.z - ls.z, rs.x - ls.x)))
    return min(yaw, 180 - yaw)


def classify_view(yaw):
    if yaw < FRONT_MAX_YAW:
        return 'front'
    if yaw > SIDE_MIN_YAW:
        return 'side'
    return 'angled'


# ------------------------------------------ FRONT VIEW ------------------------------------------ #
def front_metrics(lm, w, h):
    def px(p):
        return lm.landmark[p].x * w, lm.landmark[p].y * h

    ls, rs = px(P.LEFT_SHOULDER), px(P.RIGHT_SHOULDER)
    le, re = px(P.LEFT_EAR), px(P.RIGHT_EAR)

    shoulder_w = findDistance(*ls, *rs)
    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    ear_mid = ((le[0] + re[0]) / 2, (le[1] + re[1]) / 2)

    # Ear-to-shoulder height normalized by shoulder width, so it doesn't change with distance.
    neck_ratio = (shoulder_mid[1] - ear_mid[1]) / shoulder_w

    tilt = abs(m.degrees(m.atan2(rs[1] - ls[1], rs[0] - ls[0])))
    tilt = min(tilt, 180 - tilt)

    return {
        "ls": ls, "rs": rs, "le": le, "re": re,
        "shoulder_mid": shoulder_mid, "ear_mid": ear_mid,
        "shoulder_w": shoulder_w, "neck_ratio": neck_ratio, "tilt": tilt,
    }


def front_classify(f, base):
    neck_pct = f["neck_ratio"] / base["neck_ratio"]
    width_pct = f["shoulder_w"] / base["shoulder_w"]
    problems = []
    if neck_pct < NECK_DROP:
        problems.append("Head forward/down")
    if width_pct > LEAN_IN:
        problems.append("Leaning in")
    if f["tilt"] > SHOULDER_TILT_DEG:
        problems.append("Shoulders tilted")
    return problems, neck_pct, width_pct


def draw_front(image, f, color, lines):
    pt = lambda p: (int(p[0]), int(p[1]))
    cv2.line(image, pt(f["ls"]), pt(f["rs"]), color, 4)
    cv2.line(image, pt(f["le"]), pt(f["re"]), color, 2)
    cv2.line(image, pt(f["shoulder_mid"]), pt(f["ear_mid"]), color, 4)
    for p in ("ls", "rs", "le", "re", "shoulder_mid", "ear_mid"):
        cv2.circle(image, pt(f[p]), 6, yellow, -1)
    for i, text in enumerate(lines):
        cv2.putText(image, text, (10, 30 + 25 * i), font, 1.2, color, 2)


# ------------------------------------------ SIDE VIEW ------------------------------------------- #
def side_metrics(lm, w, h):
    def px(p):
        return int(lm.landmark[p].x * w), int(lm.landmark[p].y * h)

    def vis(*ps):
        return sum(lm.landmark[p].visibility for p in ps)

    # Use whichever side of the body faces the camera.
    if vis(P.LEFT_EAR, P.LEFT_SHOULDER) >= vis(P.RIGHT_EAR, P.RIGHT_SHOULDER):
        sh, ear, hip, far_sh = P.LEFT_SHOULDER, P.LEFT_EAR, P.LEFT_HIP, P.RIGHT_SHOULDER
    else:
        sh, ear, hip, far_sh = P.RIGHT_SHOULDER, P.RIGHT_EAR, P.RIGHT_HIP, P.LEFT_SHOULDER

    s = {"sh": px(sh), "ear": px(ear), "far_sh": px(far_sh), "hip": None, "torso": None}
    s["offset"] = findDistance(*s["sh"], *s["far_sh"])
    s["neck"] = findAngle(*s["sh"], *s["ear"])
    # Hips are often out of frame at a desk; judge on the neck alone if so.
    if lm.landmark[hip].visibility > 0.5:
        s["hip"] = px(hip)
        s["torso"] = findAngle(*s["hip"], *s["sh"])
    return s


def side_classify(s):
    return s["neck"] < NECK_MAX_DEG and (s["torso"] is None or s["torso"] < TORSO_MAX_DEG)


def draw_side(image, s, good):
    (sx, sy) = s["sh"]
    color, line_color = (light_green, green) if good else (red, red)

    for p in (s["sh"], s["ear"], (sx, sy - 100)):
        cv2.circle(image, p, 7, yellow, -1)
    cv2.circle(image, s["far_sh"], 7, pink, -1)
    cv2.line(image, s["sh"], s["ear"], line_color, 4)
    cv2.line(image, s["sh"], (sx, sy - 100), line_color, 4)
    cv2.putText(image, str(int(s["neck"])), (sx + 10, sy), font, 0.9, color, 2)

    text = 'Neck : %d' % s["neck"]
    if s["hip"] is not None:
        (hx, hy) = s["hip"]
        for p in (s["hip"], (hx, hy - 100)):
            cv2.circle(image, p, 7, yellow, -1)
        cv2.line(image, s["hip"], s["sh"], line_color, 4)
        cv2.line(image, s["hip"], (hx, hy - 100), line_color, 4)
        cv2.putText(image, str(int(s["torso"])), (hx + 10, hy), font, 0.9, color, 2)
        text += '  Torso : %d' % s["torso"]
    else:
        text += '  Torso : (hips not visible)'
    cv2.putText(image, text, (10, 30), font, 1.2, color, 2)


# -------------------------------------------- MAIN ---------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam posture monitor")
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument("--front", action="store_true", help="force front-view detection")
    view_group.add_argument("--side", action="store_true", help="force side-view detection")
    parser.add_argument("--no-preview", action="store_true",
                        help="run in the background without the video window (Ctrl+C to quit)")
    args = parser.parse_args()
    show = not args.no_preview
    forced = 'front' if args.front else 'side' if args.side else None

    cap = cv2.VideoCapture(0)

    good_frames = bad_frames = 0
    baseline, calib = None, []
    view = forced
    yaw_smooth = None
    candidate, candidate_frames = None, 0

    print("Mode:", forced or "auto (detects front or side view)")
    print("Face the camera and sit up straight for %d s to calibrate front view." % CALIBRATION_S)
    if not show:
        print("Running in background. Press Ctrl+C to quit.")

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Null.Frames")
                continue

            # Some webcams report 0 FPS; fall back to 30 to avoid dividing by zero.
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            image = cv2.flip(image, 1)  # mirror so the preview feels natural
            h, w = image.shape[:2]

            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            lm = results.pose_landmarks

            # ---- Work out which way the person is facing ----
            if forced is None and results.pose_world_landmarks is not None:
                yaw = body_yaw(results.pose_world_landmarks)
                yaw_smooth = yaw if yaw_smooth is None else 0.8 * yaw_smooth + 0.2 * yaw
                raw_view = classify_view(yaw_smooth)
                if view is None:
                    view = raw_view
                elif raw_view != view:
                    if raw_view == candidate:
                        candidate_frames += 1
                    else:
                        candidate, candidate_frames = raw_view, 1
                    if candidate_frames >= fps * VIEW_SWITCH_S:
                        view, candidate, candidate_frames = raw_view, None, 0
                        good_frames = bad_frames = 0
                        if not show:
                            print("View:", view)
                else:
                    candidate, candidate_frames = None, 0

            # ---- Judge posture with the method for that view ----
            good = None
            status = None
            if lm is None or not visible(lm, [P.LEFT_SHOULDER, P.RIGHT_SHOULDER], 0.3):
                status = ('No person detected', red)

            elif view == 'angled':
                status = ('Angled view: face the camera or turn side-on', orange)
                if show:
                    cv2.putText(image, 'Rotation %d deg' % yaw_smooth, (10, 55), font, 1.2, orange, 2)

            elif view == 'side':
                s = side_metrics(lm, w, h)
                good = side_classify(s)
                if show:
                    draw_side(image, s, good)

            elif view == 'front':
                if not visible(lm, [P.LEFT_EAR, P.RIGHT_EAR]):
                    status = ('Ears not visible', red)
                else:
                    f = front_metrics(lm, w, h)
                    if baseline is None:
                        # Calibration: average the upright pose over the first few seconds.
                        calib.append(f)
                        need = int(fps * CALIBRATION_S)
                        if len(calib) >= need:
                            baseline = {k: sum(c[k] for c in calib) / len(calib)
                                        for k in ("neck_ratio", "shoulder_w")}
                            calib = []
                            print("Calibrated.")
                        elif show:
                            left = (need - len(calib)) / fps
                            draw_front(image, f, white, ['Sit up straight: calibrating %.1fs' % left])
                    else:
                        problems, neck_pct, width_pct = front_classify(f, baseline)
                        good = not problems
                        if show:
                            lines = ['Neck %d%%  Distance %d%%  Tilt %d deg'
                                     % (neck_pct * 100, width_pct * 100, f["tilt"])]
                            lines += problems if problems else ['Good posture']
                            draw_front(image, f, green if good else red, lines)

            # ---- Time held in current posture ----
            if good is True:
                bad_frames = 0
                good_frames += 1
            elif good is False:
                good_frames = 0
                bad_frames += 1
            bad_time = bad_frames / fps

            if show:
                if status:
                    cv2.putText(image, status[0], (10, 30), font, 1.2, status[1], 2)
                if good is not None:
                    if good:
                        cv2.putText(image, 'Good Posture Time : %.1fs' % (good_frames / fps),
                                    (10, h - 20), font, 1.2, green, 2)
                    else:
                        cv2.putText(image, 'Bad Posture Time : %.1fs' % bad_time,
                                    (10, h - 20), font, 1.2, red, 2)
                label = 'View: %s%s' % (view or '...', '' if forced else ' (auto)')
                cv2.putText(image, label, (w - 200, 30), font, 1.2, white, 2)
                cv2.putText(image, "'c' recalibrate  'q' quit", (w - 260, h - 20), font, 1.0, white, 1)
                cv2.imshow(WINDOW, image)
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('c'):
                    baseline, calib = None, []
                    good_frames = bad_frames = 0

            # Alert after holding bad posture too long.
            if bad_time > BAD_POSTURE_LIMIT_S:
                sendWarning()
                bad_frames = 0
    except KeyboardInterrupt:
        pass

    cap.release()
    cv2.destroyAllWindows()