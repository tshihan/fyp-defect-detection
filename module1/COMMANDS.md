# Module 1 — Command Reference

## Setup

Install dependencies (run once from the project root):

```bash
uv pip install -r module1/requirements.txt
```

All commands below are run from inside the `module1/` directory:

```bash
cd module1
```

Place your trained weights in `module1/models/`:
- `models/pcb_best.pt` — PCB defect model
- `models/steel_best.pt` — Steel defect model

---

## Single Image Prediction

Run inference on one image and save the annotated result to `output_images/`.

```bash
python predict_single_image.py --model models/pcb_best.pt --image input_images/missing_hole.jpg --conf 0.5
```

| Argument | Description | Default |
|---|---|---|
| `--model` | Path to best.pt | required |
| `--image` | Path to input image | required |
| `--conf` | Confidence threshold (0.0 – 1.0) | `0.5` |
| `--output` | Folder to save annotated image | `output_images` |

**Example — lower threshold to catch more detections:**
```bash
python predict_single_image.py --model models/pcb_best.pt --image input_images/missing_hole.jpg --conf 0.3
```

---

## Batch Prediction

Run inference on all images in a folder. Saves annotated images and a `results.json` to `output_images/`.

```bash
python predict_batch_images.py --model models/pcb_best.pt --input input_images/pcb --conf 0.5
```

| Argument | Description | Default |
|---|---|---|
| `--model` | Path to best.pt | required |
| `--input` | Folder of input images | required |
| `--conf` | Confidence threshold | `0.5` |
| `--output` | Folder to save annotated images + results.json | `output_images` |

**Example — steel defect batch:**
```bash
python predict_batch_images.py --model models/steel_best.pt --input input_images/steel --conf 0.5
```

---

## Full Inference Pipeline

Runs batch prediction and then generates a PDF report in one command.

```bash
python run_inference_pipeline.py --model models/pcb_best.pt --input input_images/pcb --conf 0.5
```

| Argument | Description | Default |
|---|---|---|
| `--model` | Path to best.pt | required |
| `--input` | Folder of input images | `input_images` |
| `--conf` | Confidence threshold | `0.5` |
| `--output` | Folder for annotated images + results.json | `output_images` |
| `--reports` | Folder for PDF report | `reports` |
| `--skip-report` | Skip PDF generation | off |

**Example — skip the PDF report:**
```bash
python run_inference_pipeline.py --model models/pcb_best.pt --input input_images/pcb --conf 0.5 --skip-report
```

**Outputs:**
- `output_images/<name>_detected.jpg` — annotated images
- `output_images/results.json` — all detections in JSON
- `reports/DefectReport_<timestamp>.pdf` — PDF report

---

## Generate PDF Report Only

If you already have a `results.json` and only want to (re)generate the PDF:

```bash
python generate_report.py --results output_images/results.json --input input_images/pcb --output reports
```

| Argument | Description | Default |
|---|---|---|
| `--results` | Path to results.json | `output_images/results.json` |
| `--input` | Folder of original images (for side-by-side in PDF) | `input_images` |
| `--output` | Folder to save PDF | `reports` |
| `--model` | Model name shown on report cover | `YOLOv11` |

---

## Conveyor Belt Simulation

Simulates a real-time production line by sliding images through an OpenCV window one by one.

```bash
python simulate_conveyor.py --model models/pcb_best.pt --input input_images/pcb --conf 0.5 --fps 2
```

| Argument | Description | Default |
|---|---|---|
| `--model` | Path to best.pt | required |
| `--input` | Folder of images | `input_images` |
| `--conf` | Confidence threshold | `0.5` |
| `--fps` | Inspection speed (images per second) | `2.0` |

**Controls during simulation:**

| Key | Action |
|---|---|
| `q` | Quit |
| `space` | Pause / Resume |
| `s` | Save current frame to `output_images/` |
