import time
from pathlib import Path
from PIL import Image
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Defect Detection Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Section card */
        .section-card {
            background: #f8f9fb;
            border: 1px solid #dde3ec;
            border-radius: 10px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1.2rem;
        }
        /* Badge chip */
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }
        .badge-blue  { background:#dbeafe; color:#1d4ed8; }
        .badge-green { background:#dcfce7; color:#15803d; }
        .badge-amber { background:#fef9c3; color:#92400e; }

        /* Metric cards */
        div[data-testid="metric-container"] {
            background: #ffffff;
            border: 1px solid #dde3ec;
            border-radius: 10px;
            padding: 0.8rem 1rem;
        }

        /* Tighten expander header */
        .streamlit-expanderHeader { font-weight: 600; font-size: 1rem; }

        /* Make file uploader label bolder */
        div[data-testid="stFileUploader"] label { font-weight: 600; }

        /* Page‑wide top padding reduction */
        .block-container { padding-top: 1.8rem !important; }

        /* Sidebar status pills */
        .status-pill {
            display:inline-block;
            border-radius:20px;
            padding: 3px 12px;
            font-size:0.78rem;
            font-weight:600;
            margin-bottom: 6px;
        }
        .pill-green { background:#dcfce7; color:#15803d; }
        .pill-blue  { background:#dbeafe; color:#1d4ed8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ─────────────────────────────────────────────────────
for key in ("exported", "retrained", "pdf_generated"):
    if key not in st.session_state:
        st.session_state[key] = False


# ── Image helpers ──────────────────────────────────────────────────────────────
_ASSETS = Path(__file__).parent / "assets"

@st.cache_resource
def load_images():
    original = Image.open(_ASSETS / "original.png")
    detected = Image.open(_ASSETS / "detected.png")
    return original, detected


# ═══════════════════════════════════════════════════════════════════════════════
# 1 ── SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")
    st.divider()

    model_choice = st.selectbox(
        "Model Selection",
        ["YOLOv8", "YOLO11"],
        help="Select the YOLO backbone used for inference.",
    )

    conf_threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=0,
        max_value=100,
        value=75,
        step=1,
        help="Detections below this confidence score are suppressed.",
    )

    st.divider()
    st.markdown("**System Status**")
    st.markdown(
        '<span class="status-pill pill-green">🟢 API Connected · Label Studio</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="status-pill pill-green">🟢 GPU Active · CUDA 12.1</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="status-pill pill-blue">🔵 Model Loaded · ' + model_choice + "</span>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Active Run Info**")
    st.caption("Run ID: `RUN-20260506-084312`")
    st.caption("Dataset: `NEU-DET v2 (1,800 imgs)`")
    st.caption("Annotator: `Auto + HITL`")


# ═══════════════════════════════════════════════════════════════════════════════
# 2 ── HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🔍 Industrial Defect Detection & HITL Pipeline")
st.markdown(
    "**Automated Annotation and Continuous Learning** — Images are processed by a YOLO model, "
    "flagged detections are routed to Label Studio for human review, and corrected annotations "
    "trigger incremental fine-tuning to close the feedback loop."
)

uploaded_files = st.file_uploader(
    "📂 Upload Inspection Images (batch processing)",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True,
    help="Upload one or more surface-inspection images. Mock data is displayed when no files are uploaded.",
)
if uploaded_files:
    st.info(f"✅ {len(uploaded_files)} image(s) queued for inference.")

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# 3 ── INFERENCE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<span class="badge badge-blue">STAGE 1 — AUTOMATED INFERENCE</span>',
    unsafe_allow_html=True,
)
st.markdown("### 🖼️ Inference Results")

original, detected = load_images()

col_orig, col_det = st.columns(2, gap="medium")
with col_orig:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Original Image** · `sample_plate_0042.jpg`")
    st.image(original, caption="Raw input — no annotations", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_det:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f"**{model_choice} Detection Output** · conf ≥ {conf_threshold}%")
    st.image(detected, caption="3 defects detected  ·  Class: crazing (3)  ·  Conf: 91%, 83%, 76%", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")
m1, m2, m3, m4 = st.columns(4, gap="small")
with m1:
    st.metric("Total Images Processed", "12", "+4", help="Images processed in the current run")
with m2:
    st.metric(f"Defects Found (≥{conf_threshold}% conf)", "3", "+1")
with m3:
    st.metric("Processing Time", "3.42 s", "-0.8 s", delta_color="inverse")
with m4:
    st.metric("Mean Confidence Score", "81.4%", "+3.2%")

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# 4 ── HITL & RETRAINING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<span class="badge badge-amber">STAGE 2 — HUMAN-IN-THE-LOOP CORRECTION</span>',
    unsafe_allow_html=True,
)

with st.expander("🔄 Continuous Learning & Correction", expanded=True):
    st.warning(
        "⚠️ **Review Required:** Inspect the inference output above for **False Negatives** "
        "(missed defects) or **False Positives** (incorrect detections). "
        "Export flagged images to Label Studio for expert annotation before triggering retraining."
    )

    col_export, col_retrain = st.columns(2, gap="medium")

    with col_export:
        st.markdown("**Step 1 — Annotate Missed Defects**")
        st.caption("Push images with uncertain or missing predictions to Label Studio for human review.")
        if st.button("📤 Export Missed Defects to Label Studio", use_container_width=True):
            st.session_state.exported = True
            st.session_state.retrained = False

        if st.session_state.exported:
            st.info(
                "✅ **3 crazing detections** exported to Label Studio project **`DefectNet-v2`**.  \n"
                "Annotation task URL: `http://localhost:8080/projects/4/`"
            )

    with col_retrain:
        st.markdown("**Step 2 — Trigger Incremental Retraining**")
        st.caption("Fine-tune YOLO weights on the newly corrected annotation set.")
        retrain_btn = st.button(
            f"🚀 Trigger {model_choice} Retraining",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.exported,
        )

        if retrain_btn and st.session_state.exported:
            progress_bar = st.progress(0, text="Initialising training environment…")
            stages = [
                (15, "Loading corrected annotations…"),
                (30, "Augmenting dataset…"),
                (50, "Epoch 1/10 — loss: 0.4821"),
                (65, "Epoch 4/10 — loss: 0.3107"),
                (80, "Epoch 7/10 — loss: 0.2244"),
                (92, "Epoch 10/10 — loss: 0.1893"),
                (100, "Saving fine-tuned weights…"),
            ]
            for pct, msg in stages:
                time.sleep(0.45)
                progress_bar.progress(pct, text=f"⚙️ Retraining {model_choice} weights… {msg}")
            st.session_state.retrained = True

        if st.session_state.retrained:
            st.success(
                "✅ **Retraining complete!**  \n"
                "New weights saved → `/models/yolo_ft_v3.pt`  \n"
                "Validation mAP@0.5: **0.873** (+0.041 vs baseline)"
            )

    # Pipeline diagram (text-based)
    st.divider()
    st.markdown("**Continuous Learning Loop**")
    st.markdown(
        """
        ```
        Raw Images ──▶ YOLO Inference ──▶ High-Confidence Results
                              │
                              ▼ (low-conf / flagged)
                       Label Studio (Human Review)
                              │
                              ▼ Corrected Annotations
                       YOLO Fine-Tuning (Incremental)
                              │
                              ▼
                       Updated Model Weights ──▶ Re-deploy
        ```
        """
    )

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# 5 ── REPORTING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<span class="badge badge-green">STAGE 3 — REPORT GENERATION</span>',
    unsafe_allow_html=True,
)
st.markdown("### 📋 Finalize & Export")

rep_col1, rep_col2 = st.columns([2, 3], gap="medium")
with rep_col1:
    st.markdown("**Generate a structured audit report** including:")
    st.markdown(
        "- Timestamped detection log per image  \n"
        "- Bounding box coordinates & confidence scores  \n"
        "- HITL correction summary  \n"
        "- Model version & performance delta  \n"
        "- Annotator IDs and review timestamps"
    )
    if st.button("📄 Generate Automated PDF Report", type="primary", use_container_width=True):
        st.session_state.pdf_generated = True

with rep_col2:
    if st.session_state.pdf_generated:
        st.success(
            "✅ **PDF Report generated successfully** with timestamps and bounding box metrics!  \n\n"
            "**Saved to:** `./reports/DefectReport_20260506_084312.pdf`"
        )
        st.caption(
            "Report covers 12 images · 3 crazing defects · 2 HITL corrections · "
            f"Model: {model_choice} · Run ID: RUN-20260506-084312"
        )
    else:
        st.info("Click **Generate Automated PDF Report** to produce the audit document.")

st.divider()
st.caption(
    "🔍 Industrial Defect Detection & HITL Pipeline · Final Year Project · "
    "Automated Annotation and Continuous Learning for Industrial Defect Detection"
)
