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
import threading
import time
import ctypes

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════
SAMPLE_RATE    = 44100
CHUNK_DURATION = 0.05          # 50ms audio chunks (low latency)
CHUNK_SAMPLES  = int(SAMPLE_RATE * CHUNK_DURATION)

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
COLOR_NOTE_BG    = (0, 0, 0)
FONT             = cv2.FONT_HERSHEY_SIMPLEX

# ═══════════════════════════════════════════════════════════════════
#  MUSIC UTILITIES
# ═══════════════════════════════════════════════════════════════════

def build_frequency_table(octave_range):
    """Build a list of frequencies for every note in the given octave range."""
    freqs = []
    note_labels = []
    for octave in octave_range:
        for pitch in range(12):
            freq = round(440 * 2 ** ((octave - 4) + (pitch - 9) / 12), 2)
            freqs.append(freq)
            note_labels.append(f"{NOTE_NAMES[pitch]}{octave}")
    return freqs, note_labels


def build_pentatonic_table(octave_range):
    """Return only the pentatonic notes from each octave – sounds nicer."""
    freqs, labels = [], []
    for octave in octave_range:
        for semitone in PENTATONIC:
            freq = round(440 * 2 ** ((octave - 4) + (semitone - 9) / 12), 2)
            freqs.append(freq)
            labels.append(f"{NOTE_NAMES[semitone]}{octave}")
    return freqs, labels


def harmonic_from_y(frame_height, rows, yc):
    """Map the Y-position of the hand to a harmonic multiplier (1-5)."""
    section_height = int(2 * frame_height / 3) // rows
    row_index = int(yc // section_height) if section_height > 0 else 0
    harmonic = max(1, rows - row_index)
    return min(harmonic, rows)


# ═══════════════════════════════════════════════════════════════════
#  DRAWING UTILITIES
# ═══════════════════════════════════════════════════════════════════

def draw_grid(img, rows, cols, pinch_dist):
    """Draw a stylised musical-grid overlay onto the frame."""
    h, w = img.shape[:2]
    overlay = img.copy()

    # Vertical columns (pitch zones) – always shown
    col_w = w // max(cols, 1)
    for j in range(0, w, col_w):
        cv2.line(overlay, (j, 0), (j, h), COLOR_GRID, 1)

    # Horizontal rows (harmonic zones) – shown when hand is somewhat open
    if 0 < pinch_dist < 100:
        row_h = int(2 * h / 3) // rows
        for i in range(0, int(2 * h / 3) + 1, row_h):
            cv2.line(overlay, (0, i), (w, i), COLOR_GRID, 1)

    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)


def draw_hud(img, note_label, harmonic, pinch_dist, is_playing):
    """Render a heads-up-display with note name, harmonic, and status."""
    h, w = img.shape[:2]
    panel_h = 60

    # Semi-transparent bottom bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Note name (big, center)
    status_color = COLOR_PINCH_HIT if pinch_dist < 30 and pinch_dist > 0 else (
                   COLOR_PINCH_NEAR if pinch_dist < 100 and pinch_dist > 0 else COLOR_ACCENT)
    note_text = f"{note_label}  x{harmonic}" if is_playing else "-- rest --"
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
#  AUDIO ENGINE  (phase-continuous; runs in a separate Process)
# ═══════════════════════════════════════════════════════════════════

def music_generator(coord_queue, shared_freqs, shared_labels, num_notes):
    """
    Phase-continuous stereo sine-wave synthesizer.

    Key design choices:
    • We maintain `phase` across chunks – no wave restarts → no pops.
    • Frequency glides smoothly (lerp) to the target each chunk.
    • Amplitude fades in/out based on whether a note is 'active'.
    • We use a dedicated pygame Channel to avoid sound object juggling.
    """
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
    channel = pygame.mixer.Channel(0)

    phase          = 0.0
    current_freq   = 440.0
    target_freq    = 440.0
    amplitude      = 0.0        # 0.0 → 1.0
    target_amp     = 0.0

    # Local copies from shared memory
    freqs  = list(shared_freqs[:num_notes])
    labels = list(shared_labels)

    prev_xi         = 0
    prev_lenti      = 0.0
    frame_width     = 640
    frame_height    = 480
    harmonic_mult   = 1

    t_chunk = np.arange(CHUNK_SAMPLES) / SAMPLE_RATE   # time axis per chunk

    while True:
        # ── Pull latest data from queue ──────────────────────────────
        new_data = None
        while not coord_queue.empty():
            try:
                new_data = coord_queue.get_nowait()
            except Exception:
                break

        if new_data is not None:
            xi, yc, lenti, frame_width, frame_height, note_idx = new_data
            prev_lenti = lenti

            if 0 < lenti < 30:
                # Full pinch – hold fixed pitch (X locked)
                target_freq  = freqs[note_idx]
                harmonic_mult = harmonic_from_y(frame_height, 6, yc)
                target_amp   = 0.85
            elif 30 <= lenti < 100:
                # Near-pinch zone – play with live X
                target_freq  = freqs[note_idx]
                harmonic_mult = harmonic_from_y(frame_height, 6, yc)
                target_amp   = 0.65
            else:
                # Hand open – silence
                target_amp  = 0.0
                harmonic_mult = 1
        else:
            # No new data yet; gradually fade if open hand
            if prev_lenti > 100 or prev_lenti == 0:
                target_amp = 0.0

        # ── Smooth frequency glide ─────────────────────────────────--
        alpha = 0.25                                    # interpolation speed
        current_freq = current_freq + alpha * (target_freq - current_freq)
        amplitude    = amplitude    + alpha * (target_amp  - amplitude)

        eff_freq = current_freq * harmonic_mult

        # ── Generate phase-continuous chunk ──────────────────────────
        phases = 2 * np.pi * eff_freq * t_chunk + phase
        wave   = (amplitude * 32767 * np.sin(phases)).astype(np.int16)

        # Advance phase to avoid discontinuity at chunk boundary
        phase = (phase + 2 * np.pi * eff_freq * CHUNK_DURATION) % (2 * np.pi)

        # ── Play chunk ───────────────────────────────────────────────
        stereo = np.column_stack((wave, wave))
        sound  = pygame.sndarray.make_sound(stereo)

        # Queue sound so playback is seamless
        if not channel.get_busy():
            channel.play(sound)
        else:
            channel.queue(sound)

        # Sleep just under chunk duration to keep queue primed
        time.sleep(CHUNK_DURATION * 0.8)


# ═══════════════════════════════════════════════════════════════════
#  CAMERA + HAND TRACKING  (main video process)
# ═══════════════════════════════════════════════════════════════════

def camera_input(coord_queue, shared_freqs, shared_labels, num_notes):
    """Capture webcam, detect hand, and send coordinates to the audio process."""
    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.8
    )
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    freqs  = list(shared_freqs[:num_notes])
    labels = list(shared_labels)

    prev_note_idx = 0
    lenti = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        lenti = 0.0
        is_playing = False
        note_label = "--"
        harmonic   = 1
        note_idx   = prev_note_idx

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                lm = hand_lm.landmark

                # Index fingertip (8) and thumb tip (4)
                ix = int(lm[8].x * frame_w)
                iy = int(lm[8].y * frame_h)
                tx = int(lm[4].x * frame_w)
                ty = int(lm[4].y * frame_h)

                midpoint = ((ix + tx) / 2, (iy + ty) / 2)
                lenti    = float(np.hypot(abs(ix - tx), abs(iy - ty)))

                # Map mirrored X → note index
                # Flip x because we mirror the frame for display
                mirrored_ix = frame_w - ix
                note_idx = min(
                    int(mirrored_ix / frame_w * num_notes),
                    num_notes - 1
                )
                note_idx   = max(note_idx, 0)
                prev_note_idx = note_idx
                note_label = labels[note_idx]
                harmonic   = harmonic_from_y(frame_h, 6, midpoint[1])
                is_playing = lenti < 100 and lenti > 0

                coord_queue.put((ix, midpoint[1], lenti, frame_w, frame_h, note_idx))

                # Draw enhanced hand visuals
                draw_hand_visuals(frame, (ix, iy), (tx, ty), midpoint, lenti)

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
        draw_hud(display, note_label, harmonic, lenti, is_playing)

        cv2.imshow('Hand2Music 🎵', display)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    mp_proc.set_start_method('spawn', force=True)

    # ── Build note table ────────────────────────────────────────────
    octave_range = range(3, 6)                  # C3 → B5 (3 octaves)
    freqs, labels = build_pentatonic_table(octave_range)   # Pentatonic = nice
    num_notes = len(freqs)

    # Shared memory arrays
    shared_freqs  = mp_proc.Array('d', freqs)
    shared_labels = mp_proc.Array(ctypes.c_char_p, [l.encode() for l in labels])

    coord_queue = mp_proc.Queue(maxsize=10)

    # ── Spawn processes ─────────────────────────────────────────────
    p_music  = mp_proc.Process(
        target=music_generator,
        args=(coord_queue, shared_freqs, shared_labels, num_notes),
        daemon=True
    )
    p_camera = mp_proc.Process(
        target=camera_input,
        args=(coord_queue, shared_freqs, shared_labels, num_notes)
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
