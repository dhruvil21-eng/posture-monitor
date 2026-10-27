# Posture Monitor

A webcam-based posture monitor built with OpenCV and MediaPipe Pose. It watches how you sit while you work, figures out whether you're facing the camera or sitting side-on, and pops up an alert over whatever you're doing if you've been slouching for too long.

## Why I built this

I'm a Mechatronics Engineering student at McMaster, which mostly means a lot of hours at a desk: problem sets, CAD, debugging code, studying for midterms. Somewhere in there I realized I'd look up after a couple of hours and be completely hunched over my laptop with my neck aching, and I never noticed it happening. Reminders and "sit up straight" notes didn't work, because the whole problem is that you don't notice when you're slouching.

So I wanted something that would notice for me. I first built a basic version in high school that only worked with the camera placed to my side. Coming back to it, I realized nobody actually sets up a camera next to their desk, so I rebuilt it to work with a normal laptop webcam facing you, added automatic detection of which way you're sitting, and made the alert show up on top of any app instead of getting buried behind my browser.

## What it does

- Works with a normal front-facing laptop webcam (no special setup)
- Calibrates to *your* upright posture in the first 3 seconds
- Detects head drooping forward, leaning in toward the screen, and uneven shoulders
- Automatically switches to side-view angle tracking if you turn sideways
- After 20 seconds of continuous bad posture, shows a system alert over whatever app you're in
- Can run in the background with no video window

## How it works

**View detection (auto)**

MediaPipe's 3D landmarks estimate how far your shoulders are rotated from the camera. Under 30° it uses the front-view method, over 60° it uses the side-view method, and in between it asks you to turn, since neither method is reliable there. The view has to hold for 1 second before switching, so it doesn't flicker.

**Front view**

A single camera can't directly measure how far your head leans toward it, so instead the program tracks things that change when you slouch, relative to your calibrated baseline:

- **Head forward/down:** ear-to-shoulder height (divided by shoulder width so it doesn't depend on distance) drops below 80% of baseline.
- **Leaning in:** shoulders appear more than 15% wider than baseline, meaning you've moved toward the camera.
- **Shoulders tilted:** the shoulder line tilts more than 10°.

**Side view**

Neck inclination (shoulder-to-ear line) and torso inclination (hip-to-shoulder line) measured from vertical, using whichever side of the body faces the camera. If your hips are out of frame, it judges on the neck alone.

**Alert**

Bad frames are converted to time using the camera FPS. After 20 s, a system alert pops up on top of whatever app you're using (native dialog on macOS, message box on Windows, `zenity` dialog or `notify-send` notification on Linux). If none of those work, it falls back to an OpenCV window showing `alert.png` or a red warning screen.

## Run it

Requires Python 3.9 to 3.12 (MediaPipe's `solutions` API used here isn't available on newer versions).

```
pip install -r requirements.txt
python posture_monitor.py
```

Sit up straight facing the webcam while it calibrates. Press `c` to recalibrate, `q` to quit.

| Flag | What it does |
|---|---|
| `--no-preview` | Runs in the background with no video window; only the alert pops up. `Ctrl+C` to stop. |
| `--front` | Forces front-view detection instead of auto. |
| `--side` | Forces side-view detection instead of auto. |

Thresholds are constants at the top of `posture_monitor.py` if you want to make it stricter or looser.

On macOS, the first alert will ask permission to control "System Events". Allow it so the alert can appear over other apps.

## What I'd do next

- A browser version so anyone can try it from a link, no install
- Port to MediaPipe's newer Tasks API so it runs on current Python versions
- Log posture over a work session and show a summary of when you slouch most

## Acknowledgments

The side-view angle calculation and overlay approach were adapted from the LearnOpenCV tutorial on building a body posture analysis system with MediaPipe. The front-view detection, calibration, auto view switching, and cross-platform alerts are my own additions.
