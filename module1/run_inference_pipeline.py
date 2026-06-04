import argparse
import sys
from pathlib import Path

from predict_batch_images import predict_batch
from generate_report import generate_report


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end defect detection pipeline: batch inference → PDF report."
    )
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--input", default="input_images", help="Input folder of images")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--output", default="output_images", help="Folder for annotated images + results.json")
    parser.add_argument("--reports", default="reports", help="Folder for PDF report output")
    parser.add_argument("--skip-report", action="store_true", help="Skip PDF report generation")
    args = parser.parse_args()

    _banner("Industrial Defect Detection — Inference Pipeline")
    print(f"  Model  : {args.model}")
    print(f"  Input  : {args.input}")
    print(f"  Conf   : {args.conf}")
    print(f"  Output : {args.output}")
    print("=" * 58)

    # ── Step 1: Batch inference ─────────────────────────────────────────────────
    print("\n[1/2] Batch inference...")
    results = predict_batch(args.model, args.input, args.conf, args.output)

    if not results:
        print("No images processed. Place images in the input folder and retry.")
        sys.exit(1)

    # ── Step 2: PDF report ──────────────────────────────────────────────────────
    report_path = None
    if not args.skip_report:
        print("\n[2/2] Generating PDF report...")
        model_label = Path(args.model).stem
        report_path = generate_report(results, args.input, args.reports, model_label)
    else:
        print("\n[2/2] PDF report skipped (--skip-report).")

    # ── Summary ─────────────────────────────────────────────────────────────────
    total_defects = sum(len(r["detections"]) for r in results)
    _banner("Pipeline complete")
    print(f"  Images processed : {len(results)}")
    print(f"  Defects found    : {total_defects}")
    print(f"  Results JSON     : {args.output}/results.json")
    if report_path:
        print(f"  PDF report       : {report_path}")
    print("=" * 58)


def _banner(title):
    print("\n" + "=" * 58)
    print(f"  {title}")
    print("=" * 58)


if __name__ == "__main__":
    main()
