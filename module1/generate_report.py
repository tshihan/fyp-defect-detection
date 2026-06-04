import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from fpdf import FPDF


class _DefectReportPDF(FPDF):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, "Industrial Defect Detection - Automated Inspection Report", align="C")
        self.ln(2)
        self.set_draw_color(210, 210, 210)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
        self.set_text_color(30, 30, 30)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(
            0, 10,
            f"Page {self.page_no()} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            align="C",
        )


def _table_header(pdf, cols):
    pdf.set_fill_color(235, 235, 235)
    pdf.set_font("Helvetica", "B", 9)
    for label, width in cols:
        pdf.cell(width, 6, label, border=1, fill=True, align="C")
    pdf.ln()


def generate_report(results, input_dir, output_dir, model_name="YOLOv11"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf = _DefectReportPDF(model_name)
    pdf.set_margins(10, 15, 10)

    # ── Cover page ───────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.ln(18)
    pdf.cell(0, 12, "Industrial Defect Detection", align="C", ln=True)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 8, "Automated Inspection Report", align="C", ln=True)
    pdf.ln(12)

    total_images = len(results)
    total_defects = sum(len(r["detections"]) for r in results)
    all_classes = [d["class"] for r in results for d in r["detections"]]
    class_counts = Counter(all_classes)
    avg_conf = (
        sum(d["confidence"] for r in results for d in r["detections"]) / total_defects
        if total_defects else 0
    )

    summary_rows = [
        ("Run Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Model", model_name),
        ("Images Processed", str(total_images)),
        ("Total Defects Found", str(total_defects)),
        ("Mean Confidence", f"{avg_conf:.1%}"),
        ("Classes Detected", ", ".join(class_counts) if class_counts else "None"),
    ]
    pdf.set_text_color(30, 30, 30)
    for label, value in summary_rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 7, label + ":", border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, value, border=0, ln=True)

    if class_counts:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Defect Class Distribution", ln=True)
        _table_header(pdf, [("Class", 80), ("Count", 35), ("% of Total", 45)])
        pdf.set_font("Helvetica", "", 9)
        for cls, cnt in class_counts.most_common():
            pct = cnt / total_defects * 100
            pdf.cell(80, 5, cls, border=1)
            pdf.cell(35, 5, str(cnt), border=1, align="C")
            pdf.cell(45, 5, f"{pct:.1f}%", border=1, align="C")
            pdf.ln()

    # ── Per-image pages ──────────────────────────────────────────────────────────
    for r in results:
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 7, f"Image: {r['filename']}", ln=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(110, 110, 110)
        pdf.cell(
            0, 5,
            f"Timestamp: {r['timestamp']}   |   Processing time: {r['processing_time_s']}s"
            f"   |   Defects found: {len(r['detections'])}",
            ln=True,
        )
        pdf.ln(2)

        orig_path = input_dir / r["filename"]
        det_path = Path(r["output_path"])
        IMG_W, IMG_H = 90, 68
        y_img = pdf.get_y()

        if orig_path.exists():
            pdf.image(str(orig_path), x=10, y=y_img, w=IMG_W, h=IMG_H)
        if det_path.exists():
            pdf.image(str(det_path), x=108, y=y_img, w=IMG_W, h=IMG_H)

        pdf.set_y(y_img + IMG_H + 1)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(130, 130, 130)
        pdf.cell(IMG_W + 8, 5, "Original", align="C")
        pdf.cell(IMG_W, 5, "Detected", align="C")
        pdf.ln(5)

        pdf.set_text_color(30, 30, 30)
        if r["detections"]:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Detections", ln=True)
            _table_header(pdf, [("Class", 52), ("Confidence", 32), ("Bounding Box (x1, y1, x2, y2)", 96)])
            pdf.set_font("Helvetica", "", 9)
            for d in r["detections"]:
                bbox = "  ".join(str(v) for v in d["bbox_xyxy"])
                pdf.cell(52, 5, d["class"], border=1)
                pdf.cell(32, 5, f"{d['confidence']:.1%}", border=1, align="C")
                pdf.cell(96, 5, bbox, border=1, align="C")
                pdf.ln()
        else:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(150, 150, 150)
            pdf.cell(0, 6, "No defects detected above confidence threshold.", ln=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "PCB" if "pcb" in model_name.lower() else "Steel"
    out_path = output_dir / f"{prefix}_DefectReport_{ts}.pdf"
    pdf.output(str(out_path))
    print(f"PDF report saved → {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate PDF report from batch inference results.")
    parser.add_argument("--results", default="output_images/results.json", help="Path to results.json")
    parser.add_argument("--input", default="input_images", help="Folder containing original images")
    parser.add_argument("--output", default="reports", help="Output folder for PDF")
    parser.add_argument("--model", default="YOLOv11", help="Model name shown on report cover")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run predict_batch_images.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    generate_report(results, args.input, args.output, args.model)


if __name__ == "__main__":
    main()
