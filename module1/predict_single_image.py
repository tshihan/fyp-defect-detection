import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def predict_single(model_path, image_path, conf, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    results = model.predict(str(image_path), conf=conf, save=False, verbose=False)

    result = results[0]
    img_path = Path(image_path)
    out_path = output_dir / f"{img_path.stem}_detected{img_path.suffix}"
    cv2.imwrite(str(out_path), result.plot())

    detections = [
        {
            "class": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bbox_xyxy": [round(v, 1) for v in box.xyxy[0].tolist()],
        }
        for box in result.boxes
    ]

    print(f"\nImage : {image_path}")
    print(f"Output: {out_path}")
    print(f"Found {len(detections)} detection(s):")
    for d in detections:
        print(f"  {d['class']:<15}  conf={d['confidence']:.1%}  bbox={d['bbox_xyxy']}")

    return detections, str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on a single image.")
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--output", default="output_images", help="Output folder (default: output_images)")
    args = parser.parse_args()

    predict_single(args.model, args.image, args.conf, args.output)


if __name__ == "__main__":
    main()
