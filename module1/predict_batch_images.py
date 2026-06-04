import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def predict_batch(model_path, input_dir, conf, output_dir):
    input_dir = Path(input_dir)
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in {input_dir}")
        return []

    model = YOLO(model_path)
    names = model.names
    is_pcb = "pcb" in Path(model_path).stem.lower()

    print(f"\nProcessing {len(image_paths)} image(s)  |  model={model_path}  |  conf>={conf}\n")
    print(f"{'Image':<35} {'Defects':>7}  {'Classes Found'}")
    print("-" * 70)

    all_results = []
    for img_path in image_paths:
        t0 = time.perf_counter()
        img = cv2.imread(str(img_path))
        if is_pcb:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_for_model = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            img_for_model = img
        result = model.predict(img_for_model, conf=conf, save=False, verbose=False)[0]
        elapsed = time.perf_counter() - t0

        out_path = output_dir / f"{img_path.stem}_detected{img_path.suffix}"
        cv2.imwrite(str(out_path), result.plot(img=img))

        detections = [
            {
                "class": names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "bbox_xyxy": [round(v, 1) for v in box.xyxy[0].tolist()],
            }
            for box in result.boxes
        ]
        classes_found = ", ".join(sorted({d["class"] for d in detections})) or "-"

        all_results.append({
            "filename": img_path.name,
            "output_path": str(out_path),
            "timestamp": datetime.now().isoformat(),
            "processing_time_s": round(elapsed, 3),
            "detections": detections,
        })

        print(f"{img_path.name:<35} {len(detections):>7}  {classes_found}  ({elapsed:.2f}s)")

    total_defects = sum(len(r["detections"]) for r in all_results)
    print("-" * 70)
    print(f"Done: {len(image_paths)} images | {total_defects} total defects\n")

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on a folder of images.")
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--input", required=True, help="Input folder containing images")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--output", default="output_images", help="Output folder (default: output_images)")
    args = parser.parse_args()

    predict_batch(args.model, args.input, args.conf, args.output)


if __name__ == "__main__":
    main()
