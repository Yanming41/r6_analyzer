"""
Generate a synthetic test video that simulates Rainbow Six Siege HUD elements.
Requires only OpenCV – no GPU needed.

Usage:
    python create_test_video.py              # creates test_clip.mp4 (30 s)
    python create_test_video.py --duration 60
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def draw_r6_frame(
    frame: np.ndarray,
    t: float,
    events: list,
    fps: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    f = frame.copy()

    # ── Dark background ────────────────────────────────────────────────────────
    f[:] = (15, 15, 20)

    # ── Fake minimap (top-right corner) ───────────────────────────────────────
    cv2.rectangle(f, (w - 180, 10), (w - 10, 130), (30, 30, 40), -1)
    cv2.rectangle(f, (w - 180, 10), (w - 10, 130), (80, 80, 90), 1)
    cv2.putText(f, "MINIMAP", (w - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 130), 1)

    # ── Timer (top-center) ────────────────────────────────────────────────────
    remaining = max(0, 180 - int(t % 180))
    timer_str = f"{remaining // 60}:{remaining % 60:02d}"
    cv2.putText(f, timer_str, (w // 2 - 35, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 2)

    # ── Health bar (bottom-left) ───────────────────────────────────────────────
    cv2.putText(f, "HP", (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.rectangle(f, (50, h - 72), (200, h - 55), (50, 50, 50), -1)
    cv2.rectangle(f, (50, h - 72), (150, h - 55), (40, 180, 40), -1)

    # ── Score (top-left) ──────────────────────────────────────────────────────
    cv2.putText(f, "ATK 2 – 3 DEF", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1)

    # ── Operator icon placeholder ──────────────────────────────────────────────
    cv2.circle(f, (w // 2, h // 2), 8, (60, 60, 200), -1)

    # ── Active event overlay ──────────────────────────────────────────────────
    for ev_time, ev_type, ev_desc in events:
        if abs(t - ev_time) < 2.0:  # show overlay for 2 s around event
            colour = {
                "KILL": (50, 220, 50),
                "DEATH": (50, 50, 220),
                "MULTI_KILL": (50, 220, 220),
                "CLUTCH": (220, 180, 50),
                "ROUND_START": (200, 200, 200),
                "ROUND_END": (200, 200, 200),
            }.get(ev_type, (200, 200, 200))
            text = f">>> {ev_type}: {ev_desc}"
            cv2.putText(f, text, (40, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

    # ── Timestamp watermark ────────────────────────────────────────────────────
    ts = f"{int(t // 60):02d}:{int(t % 60):02d}"
    cv2.putText(f, ts, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=30, help="Video length in seconds")
    ap.add_argument("--output", default="test_clip.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    # Scripted events at specific timestamps
    EVENTS = [
        (4.0,  "ROUND_START", "Attack round begins"),
        (8.5,  "KILL",        "Headshot on Rook in kitchen"),
        (13.0, "KILL",        "Flanked Echo near obj"),
        (13.8, "MULTI_KILL",  "Double kill at window"),
        (20.0, "DEATH",       "Player down by Kapkan trap"),
        (26.0, "ROUND_END",   "Attack team wins"),
    ]

    out_path = Path(args.output)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))

    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    total_frames = args.duration * args.fps

    print(f"Generating {args.duration}s test video → {out_path}")
    for i in range(total_frames):
        t = i / args.fps
        f = draw_r6_frame(frame, t, EVENTS, args.fps)
        writer.write(f)
        if i % (args.fps * 5) == 0:
            print(f"  {t:.0f}s / {args.duration}s …")

    writer.release()
    print(f"\nDone! Saved to: {out_path.resolve()}")
    print("\nExpected events in this clip:")
    for ev_t, ev_type, ev_desc in EVENTS:
        m, s = int(ev_t // 60), int(ev_t % 60)
        print(f"  [{m:02d}:{s:02d}] - {ev_type} - {ev_desc}")


if __name__ == "__main__":
    main()
