# Hand2Music

A real-time **hand-tracking musical theremin** built in Python.  
Use your webcam to detect hand gestures and generate smooth sine-wave audio.

## What it does

- **Left ↔ Right movement** controls **pitch** (note selection across a scale)
- **Up ↓ Down movement** controls **harmonic / octave multiplier**
- **Pinch (index + thumb)** controls **playing**
  - **< 30px**: hold / sustained note
  - **30–100px**: play (near-pinch)
  - **> 100px**: silence
- Press **ESC** to quit

The recommended “final demo” script is: **`mn7.py`**.

---

## Requirements

This project uses:

- Python 3.x
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)
- Pygame (`pygame`)

---

## Installation

Create a virtual environment (optional but recommended), then install dependencies:

```bash
pip install opencv-python mediapipe numpy pygame
```

> Note: Camera permissions may be required on macOS/Windows.

---

## Run

Start the main demo:

```bash
python mn7.py
```

You should see a webcam window. Use the gestures above to play sound.

---

## Project structure (quick guide)

- `mn7.py` — **final refined demo** (recommended entry point)
- `pygame_musicgen.py` — audio generation experiment / helper
- `mn.py`, `mn1.py`, `mn2.py`, `mn4.py`, `mn5.py`, `mn6.py` — earlier iterations and experiments
- `mnp1.py` — alternate base attempt (not recommended)
- `notes/` — output folder used by earlier scripts (e.g., `.wav` generation)

---

## Troubleshooting

**No camera feed / black window**
- Ensure no other app is using the webcam.
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.

**No audio output**
- Make sure your audio device is available and not muted.
- Some systems require installing/repairing SDL audio backend (used by pygame).

**Lag / choppy audio**
- Close other heavy apps.
- Try reducing camera resolution inside `mn7.py` (advanced users).

---

## Credits

- `hrshvs`: Repository Owner 
- `CoderLakshyaYadav`: added smooth audio synthesis, pentatonic scale, enhanced UI
---
