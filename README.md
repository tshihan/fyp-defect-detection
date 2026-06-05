# Automated Annotation and Continuous Learning Pipeline for Industrial Defect Detection

Final Year Project — end-to-end pipeline that combines YOLO-based defect detection, human-in-the-loop annotation, active learning, and incremental model retraining for industrial surface inspection.

---

## Project Status

| Module | Description | Status |
|--------|-------------|--------|
| **Module 1** | Automated inference — batch prediction, conveyor simulation, PDF reporting | ✅ Complete |
| **Module 2** | Human-in-the-loop correction — Label Studio annotation workflow | 🚧 Not started |
| **Module 3** | Report generation — audit reports, detection logs, performance summaries | 🚧 Not started |

---

## Module 1 — What's Implemented

### Models
Two trained YOLO models are included:

| Model | File | Dataset | Defect Classes |
|-------|------|---------|----------------|
| PCB | `module1/models/pcb_best.pt` | PCB defect dataset | missing hole, mouse bite, open circuit, short, spur, spurious copper |
| Steel Surface | `module1/models/steel_best.pt` | NEU surface defect dataset | crazing, inclusion, patches, pitted surface, rolled-in scale, scratches |

### Scripts (`module1/`)

| Script | What it does |
|--------|-------------|
| `predict_single_image.py` | Run inference on a single image |
| `predict_batch_images.py` | Run inference on a folder of images, save annotated outputs + `results.json` |
| `simulate_conveyor.py` | OpenCV window simulation of a real-time conveyor belt inspection line |
| `run_inference_pipeline.py` | End-to-end: batch inference → PDF audit report in one command |
| `generate_report.py` | Generate a PDF report from an existing `results.json` |

### Streamlit UI (`app.py`)
Three tabs under **Module 1**:

- **Predict Batch Images** — upload custom images or use built-in datasets, view annotated results grid with metrics
- **Simulate Conveyor** — live frame-by-frame inference with adjustable speed and PASS/FAIL status
- **Run Inference Pipeline** — full batch inference + PDF report with in-browser download

---

## What's Not Yet Implemented

### Module 2 — Human-in-the-Loop Correction
- Label Studio integration for routing low-confidence / flagged detections to human annotators
- Expert review queue and annotation task management
- Corrected annotation export back into the training pipeline

### Module 3 — Report Generation
- Automated audit reports covering detection logs and bounding box exports
- Model performance summaries and version comparisons
- Annotator IDs and review timestamps

---

## Setup

### Prerequisites
- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### Install Dependencies

```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## CLI Usage (Module 1)

All commands below are run from inside `module1/`:

```bash
cd module1
```

### Single Image Prediction
```bash
python predict_single_image.py --model models/pcb_best.pt --image input_images/pcb/<image>.jpg --conf 0.5
```

### Batch Prediction
```bash
# PCB
python predict_batch_images.py --model models/pcb_best.pt --input input_images/pcb --conf 0.5

# Steel
python predict_batch_images.py --model models/steel_best.pt --input input_images/steel --conf 0.5
```

### Conveyor Simulation
```bash
python simulate_conveyor.py --model models/pcb_best.pt --input input_images/pcb --conf 0.5 --fps 2
```

Window controls: `space` pause/resume · `s` save frame · `q` quit

### Full Inference Pipeline
```bash
# PCB
python run_inference_pipeline.py --model models/pcb_best.pt --input input_images/pcb --conf 0.5

# Steel
python run_inference_pipeline.py --model models/steel_best.pt --input input_images/steel --conf 0.5
```

Outputs saved to `module1/output_images/<timestamp>/` and `module1/reports/`.

---

## Project Structure

```
fyp-defect-detection/
├── app.py                        # Streamlit UI
├── requirements.txt              # Project dependencies
└── module1/
    ├── models/
    │   ├── pcb_best.pt
    │   └── steel_best.pt
    ├── input_images/
    │   ├── pcb/                  # Built-in PCB test images
    │   └── steel/                # Built-in steel test images
    ├── output_images/            # Annotated output images + results.json
    ├── reports/                  # Generated PDF reports
    ├── predict_single_image.py
    ├── predict_batch_images.py
    ├── simulate_conveyor.py
    ├── run_inference_pipeline.py
    ├── generate_report.py
    └── COMMANDS.md               # Full CLI reference
```
