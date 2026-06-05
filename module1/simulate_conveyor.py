import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

_BELT_BG = (40, 40, 40)
_BELT_LINE = (60, 60, 60)
_COLOR_PASS = (0, 200, 80)
_COLOR_FAIL = (0, 60, 230)

DISPLAY_W, DISPLAY_H = 960, 580
SLIDE_STEPS = 10
HUD_TOP_H = 38
HUD_BOT_H = 28


def _belt_background():
    canvas = np.full((DISPLAY_H, DISPLAY_W, 3), _BELT_BG, dtype=np.uint8)
    for y in range(0, DISPLAY_H, 22):
        cv2.line(canvas, (0, y), (DISPLAY_W, y), _BELT_LINE, 1)
    return canvas


def _slide_frames(prev, curr):
    """Yield composite frames that slide curr in from the right."""
    h, w = curr.shape[:2]
    for i in range(1, SLIDE_STEPS + 1):
        offset = int(w * i / SLIDE_STEPS)
        frame = np.empty_like(curr)
        frame[:, : w - offset] = prev[:, offset:]
        frame[:, w - offset :] = curr[:, :offset]
        yield frame


def _draw_hud(frame, idx, total, filename, defect_count, conf, fps):
    out = frame.copy()

    # Top bar
    cv2.rectangle(out, (0, 0), (DISPLAY_W, HUD_TOP_H), (18, 18, 18), -1)
    cv2.putText(
        out, f"[{idx}/{total}] {filename}",
        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
    )
    status_text = f"DEFECT DETECTED  ({defect_count})" if defect_count else "PASS"
    status_color = _COLOR_FAIL if defect_count else _COLOR_PASS
    cv2.putText(
        out, status_text,
        (DISPLAY_W - 300, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA,
    )

    # Bottom bar
    bot_y = DISPLAY_H - HUD_BOT_H
    cv2.rectangle(out, (0, bot_y), (DISPLAY_W, DISPLAY_H), (18, 18, 18), -1)
    cv2.putText(
        out,
        f"conf>={conf}  |  FPS={fps}  |  [q] quit  [s] save frame  [space] pause/resume",
        (10, DISPLAY_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (150, 150, 150), 1, cv2.LINE_AA,
    )

    return out


def run_conveyor(model_path, input_dir, conf, fps):
    input_dir = Path(input_dir)
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    model = YOLO(model_path)
    hold_ms = max(1, int(1000 / fps))

    print(f"\nConveyor simulation: {len(image_paths)} images at {fps} FPS")
    print("Controls: [q] quit  [s] save frame  [space] pause/resume\n")

    win = "Conveyor Belt Simulation — press q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISPLAY_W, DISPLAY_H)

    prev_frame = _belt_background()
    paused = False

    for idx, img_path in enumerate(image_paths, start=1):
        result = model.predict(str(img_path), conf=conf, save=False, verbose=False)[0]
        annotated = cv2.resize(result.plot(), (DISPLAY_W, DISPLAY_H))
        defect_count = len(result.boxes)

        # Slide-in animation
        for slide in _slide_frames(prev_frame, annotated):
            cv2.imshow(win, _draw_hud(slide, idx, len(image_paths), img_path.name, defect_count, conf, fps))
            if cv2.waitKey(30) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return

        # Hold frame for (hold_ms) ms, supporting pause and save
        display = _draw_hud(annotated, idx, len(image_paths), img_path.name, defect_count, conf, fps)
        t_start = time.perf_counter()
        while True:
            cv2.imshow(win, display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                return

            if key == ord("s"):
                save_path = input_dir.parent / "output_images" / f"conveyor_{idx:04d}_{img_path.stem}.jpg"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), annotated)
                print(f"Saved: {save_path}")

            if key == ord(" "):
                paused = not paused
                t_start = time.perf_counter()  # reset hold timer when resuming

            if not paused and (time.perf_counter() - t_start) * 1000 >= hold_ms:
                break

        prev_frame = annotated

    cv2.destroyAllWindows()
    print("Simulation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate real-time conveyor belt inspection from a folder of images."
    )
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--input", default="input_images", help="Folder of images (default: input_images)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--fps", type=float, default=2.0, help="Inspection speed in images/sec (default: 2)")
    args = parser.parse_args()

    run_conveyor(args.model, args.input, args.conf, args.fps)


if __name__ == "__main__":
    main()
