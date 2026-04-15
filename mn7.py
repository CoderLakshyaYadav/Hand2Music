"""
Hand2Music - mn7.py
====================
A real-time hand-tracking musical theremin using MediaPipe, OpenCV, and Pygame.

Controls:
  - Move hand LEFT ↔ RIGHT  : Change note (X axis = pitch across the scale)
  - Move hand UP ↓ DOWN     : Change harmonic (octave/harmonic multiplier)
  - PINCH index + thumb     : Activate note (< 30px gap = hold, 30-100px = play)
  - Press ESC               : Quit
"""

import cv2
import numpy as np
import pygame
import mediapipe as mp
import multiprocessing as mp_proc

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Pentatonic scale semitone offsets (sounds pleasing, no "wrong" notes)
PENTATONIC = [0, 2, 4, 7, 9]

# Visual Colors (BGR)
COLOR_GRID       = (40, 40, 80)
COLOR_ACCENT     = (0, 200, 255)      # Cyan
COLOR_PINCH_FULL = (50, 255, 50)      # Green   – open hand / playing
COLOR_PINCH_NEAR = (50, 200, 255)     # Cyan    – near pinch (30-100px)
COLOR_PINCH_HIT  = (0, 80, 255)       # Red     – full pinch (<30px)
COLOR_LINE       = (0, 80, 255)
FONT             = cv2.FONT_HERSHEY_SIMPLEX


# ═══════════════════════════════════════════════════════════════════
#  MUSIC UTILITIES
# ═══════════════════════════════════════════════════════════════════

def build_pentatonic_table(octave_range):
    """Return only the pentatonic note frequencies from each octave."""
    freqs = []
    labels = []
    for octave in octave_range:
        for semitone in PENTATONIC:
            freq = round(440 * 2 ** ((octave - 4) + (semitone - 9) / 12), 2)
            freqs.append(freq)
            labels.append(f"{NOTE_NAMES[semitone]}{octave}")
    return freqs, labels


def harmonic(h, rows, yc):
    """Map Y position of the hand to a harmonic multiplier (1-5)."""
    ygrid = []
    for i in range(0, int(2 * h / 3) + 1, h // rows):
        ygrid.append(i)
    if len(ygrid) < 5:
        return 1
    if 0 < yc < ygrid[1]:
        return 5
    elif ygrid[1] < yc < ygrid[2]:
        return 4
    elif ygrid[2] < yc < ygrid[3]:
        return 3
    elif ygrid[3] < yc < ygrid[4]:
        return 2
    else:
        return 1


# ═══════════════════════════════════════════════════════════════════
#  DRAWING UTILITIES
# ═══════════════════════════════════════════════════════════════════

def draw_grid(img, rows, cols, lenti, color=COLOR_GRID, thickness=1):
    """Draw a stylised musical-grid overlay onto the frame."""
    h, w = img.shape[:2]
    if lenti > 30 or lenti == 0:
        col_w = w // max(cols, 1)
        for j in range(0, w, col_w):
            cv2.line(img, (j, 0), (j, h), color, thickness)
    if 0 < lenti < 100:
        row_h = h // rows
        for i in range(0, int(2 * h / 3) + 1, row_h):
            cv2.line(img, (0, i), (w, i), color, thickness)


def draw_hud(img, note_label, harmonic_val, pinch_dist, is_playing):
    """Render a heads-up-display with note name, harmonic, and status."""
    h, w = img.shape[:2]
    panel_h = 60

    # Semi-transparent bottom bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Note name (big, center)
    status_color = (COLOR_PINCH_HIT if 0 < pinch_dist < 30 else
                    COLOR_PINCH_NEAR if 0 < pinch_dist < 100 else COLOR_ACCENT)
    note_text = f"{note_label}  x{harmonic_val}" if is_playing else "-- rest --"
    (tw, _), _ = cv2.getTextSize(note_text, FONT, 1.0, 2)
    cv2.putText(img, note_text, ((w - tw) // 2, h - 18),
                FONT, 1.0, status_color, 2, cv2.LINE_AA)

    # Pinch distance indicator (top-left)
    cv2.putText(img, f"pinch: {pinch_dist:.0f}px", (10, 24),
                FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # Title (top-right)
    title = "Hand2Music"
    (tw, _), _ = cv2.getTextSize(title, FONT, 0.6, 1)
    cv2.putText(img, title, (w - tw - 10, 24),
                FONT, 0.6, COLOR_ACCENT, 1, cv2.LINE_AA)


def draw_hand_visuals(img, index_tip, thumb_tip, midpoint, pinch_dist):
    """Draw fancy visuals on top of hand landmarks."""
    ix, iy = index_tip
    tx, ty = thumb_tip
    mx, my = int(midpoint[0]), int(midpoint[1])

    if 0 < pinch_dist < 30:
        # Full pinch – red burst
        cv2.circle(img, (mx, my), 18, COLOR_PINCH_HIT, -1)
        cv2.circle(img, (mx, my), 22, COLOR_PINCH_HIT, 2)
    elif 0 < pinch_dist < 100:
        # Near pinch – cyan line and midpoint
        cv2.line(img, (ix, iy), (tx, ty), COLOR_LINE, 2)
        cv2.circle(img, (mx, my), 12, COLOR_PINCH_NEAR, -1)
        cv2.circle(img, (ix, iy), 8, COLOR_PINCH_FULL, -1)
        cv2.circle(img, (tx, ty), 8, COLOR_PINCH_FULL, -1)
    else:
        # Open hand – solid green dots
        cv2.circle(img, (ix, iy), 10, COLOR_PINCH_FULL, -1)
        cv2.circle(img, (tx, ty), 10, COLOR_PINCH_FULL, -1)

    # Glow ring around midpoint
    if 0 < pinch_dist < 100:
        radius = max(4, int(pinch_dist / 3))
        cv2.circle(img, (mx, my), radius, COLOR_ACCENT, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
#  AUDIO ENGINE  (runs in a separate Process)
# ═══════════════════════════════════════════════════════════════════

def music_generator(coord_queue, frequency):
    """
    Simple stop-and-play synthesizer (same reliable approach as oldmn7).
    Stops the previous sound and immediately plays the new one.
    """
    pygame.mixer.init()
    sample_rate = 44100
    duration = 1.0
    xp = 0
    n = 1
    plenti = None
    sound = None

    while True:
        if not coord_queue.empty():
            xi, yc, lenti, frame_width, frame_height = coord_queue.get()

            if lenti < 30 and lenti > 0:
                if not (plenti is not None and 30 < plenti < 100):
                    xi = xp
                n = harmonic(frame_height, 6, yc)
            elif 30 < lenti < 100:
                xi = xp
                plenti = lenti
            else:
                n = 1

            freq_idx = min(xi // (frame_width // len(frequency[:])), len(frequency[:]) - 1)
            freq_idx = max(freq_idx, 0)
            freq_to_play = frequency[freq_idx]

            # Generate sine wave
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = 32767 * np.sin(2 * np.pi * n * freq_to_play * t)

            # Convert 1D wave to 2D (stereo)
            stereo_wave = np.column_stack((wave, wave)).astype(np.int16)

            # Stop previous sound
            if sound is not None:
                try:
                    sound.stop()
                except Exception:
                    pass

            # Create sound object and play
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.play()

            xp = xi


# ═══════════════════════════════════════════════════════════════════
#  CAMERA + HAND TRACKING
# ═══════════════════════════════════════════════════════════════════

def camera_input(coord_queue, frequency):
    """Capture webcam, detect hand, and send coordinates to the audio process."""
    num_notes = len(frequency[:])
    # Build labels for display (pentatonic over octaves 3-5)
    labels, _ = [], []
    octave_range = range(3, 6)
    for octave in octave_range:
        for semitone in PENTATONIC:
            labels.append(f"{NOTE_NAMES[semitone]}{octave}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.8
    )
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    lenti = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        lenti = 0.0
        is_playing = False
        note_label = "--"
        harmonic_val = 1
        xi = yi = xt = yt = 0
        xc = yc = 0.0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                lm = hand_lm.landmark

                # Index fingertip (8) and thumb tip (4)
                xi = int(lm[8].x * frame_w)
                yi = int(lm[8].y * frame_h)
                xt = int(lm[4].x * frame_w)
                yt = int(lm[4].y * frame_h)

                xc, yc = (xi + xt) / 2, (yi + yt) / 2
                lenti = float(np.hypot(abs(xi - xt), abs(yi - yt)))

                # Map X → note index (using raw xi, same as oldmn7)
                note_idx = min(xi // (frame_w // num_notes), num_notes - 1)
                note_idx = max(note_idx, 0)
                note_label = labels[note_idx]
                harmonic_val = harmonic(frame_h, 6, yc)
                is_playing = 0 < lenti < 100

                coord_queue.put((xi, yc, lenti, frame_w, frame_h))

                # Draw enhanced hand visuals
                draw_hand_visuals(frame, (xi, yi), (xt, yt), (xc, yc), lenti)

                # MediaPipe skeleton
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

        # Mirror frame for natural feel
        display = cv2.flip(frame, 1)

        # Overlays
        draw_grid(display, 6, num_notes, lenti)
        draw_hud(display, note_label, harmonic_val, lenti, is_playing)

        cv2.imshow('Hand2Music 🎵', display)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Build pentatonic note table (C3 → B5, 3 octaves)
    octave_range = range(3, 6)
    freqs, labels = build_pentatonic_table(octave_range)
    num_notes = len(freqs)

    # Shared memory frequency array (same as oldmn7 pattern)
    shared_freqs = mp_proc.Array('d', freqs)

    coord_queue = mp_proc.Queue()

    # Spawn processes
    p_music = mp_proc.Process(
        target=music_generator,
        args=(coord_queue, shared_freqs),
        daemon=True
    )
    p_camera = mp_proc.Process(
        target=camera_input,
        args=(coord_queue, shared_freqs)
    )

    print("🎵  Hand2Music starting…")
    print(f"    Scale  : Pentatonic  |  Octaves : {list(octave_range)}")
    print(f"    Notes  : {num_notes}  |  Pinch index+thumb to play!")
    print("    Press ESC in the camera window to quit.\n")

    p_music.start()
    p_camera.start()

    p_camera.join()
    p_music.terminate()
    p_music.join()
    print("👋  Hand2Music stopped.")
