import sys
import time
import shutil
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── Module 1 path setup ────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
_MODULE1 = _ROOT / "module1"
sys.path.insert(0, str(_MODULE1))

from predict_batch_images import predict_batch  # noqa: E402
from generate_report import generate_report      # noqa: E402

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Defect Detection Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }
        .badge-blue  { background:#dbeafe; color:#1d4ed8; }
        .badge-gray  { background:#f3f4f6; color:#6b7280; }

        div[data-testid="metric-container"] {
            background: #ffffff;
            border: 1px solid #dde3ec;
            border-radius: 10px;
            padding: 0.8rem 1rem;
        }
        .block-container { padding-top: 1.8rem !important; }
        .status-pill {
            display: inline-block;
            border-radius: 20px;
            padding: 3px 12px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .pill-green { background:#dcfce7; color:#15803d; }
        .pill-blue  { background:#dbeafe; color:#1d4ed8; }
        .coming-soon-box {
            background: #f9fafb;
            border: 2px dashed #d1d5db;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            color: #9ca3af;
            margin-bottom: 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
_MODELS = {
    "PCB": _MODULE1 / "models" / "pcb_best.pt",
    "Steel Surface": _MODULE1 / "models" / "steel_best.pt",
}
_INPUT_DIRS = {
    "PCB": _MODULE1 / "input_images" / "pcb",
    "Steel Surface": _MODULE1 / "input_images" / "steel",
}
_OUTPUT_DIR = _MODULE1 / "output_images"
_REPORTS_DIR = _MODULE1 / "reports"
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ── Session state ──────────────────────────────────────────────────────────────
for _key in ("batch_results", "pipeline_results", "pipeline_report_path"):
    if _key not in st.session_state:
        st.session_state[_key] = None


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")
    st.divider()

    model_choice = st.selectbox(
        "Detection Model",
        list(_MODELS.keys()),
        help="PCB → pcb_best.pt · Steel Surface → steel_best.pt",
    )

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.50,
        step=0.05,
        format="%.2f",
        help="Detections below this score are suppressed.",
    )

    st.divider()
    st.markdown("**System Status**")
    st.markdown(
        f'<span class="status-pill pill-green">🟢 Model Ready · {model_choice}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="status-pill pill-blue">🔵 Module 1 Active</span>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Loaded Weights**")
    st.caption(f"`{_MODELS[model_choice].name}`")
    st.markdown("**Input Dataset**")
    st.caption(f"`input_images/{model_choice.lower().replace(' ', '_')}/`")


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🔍 Industrial Defect Detection Pipeline")
st.markdown(
    f"**Module 1 — Batch Inference & Conveyor Simulation** · "
    f"Model: `{_MODELS[model_choice].name}` · Threshold: `{conf_threshold:.0%}`"
)
st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<span class="badge badge-blue">MODULE 1 — INFERENCE PIPELINE</span>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs([
    "📦 Predict Batch Images",
    "🏭 Simulate Conveyor",
    "🔗 Run Inference Pipeline",
])


# ── Tab 1: Batch Predict ───────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Batch Image Prediction")
    st.caption(
        "Run YOLO inference on the built-in dataset or your own uploaded images. "
        "Results are saved to `module1/output_images/`."
    )

    uploaded_files = st.file_uploader(
        "Upload custom images (leave empty to use built-in dataset)",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    run_batch = st.button("▶ Run Batch Inference", type="primary", key="run_batch")

    if run_batch:
        if uploaded_files:
            tmp_dir = Path(tempfile.mkdtemp())
            for uf in uploaded_files:
                (tmp_dir / uf.name).write_bytes(uf.read())
            input_dir = tmp_dir
        else:
            input_dir = _INPUT_DIRS[model_choice]

        with st.spinner(f"Running {model_choice} inference…"):
            results = predict_batch(
                str(_MODELS[model_choice]),
                str(input_dir),
                conf_threshold,
                str(_OUTPUT_DIR),
            )

        if uploaded_files:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if results:
            st.session_state.batch_results = results
        else:
            st.error("No images found in the selected input folder.")

    if st.session_state.batch_results:
        results = st.session_state.batch_results
        total_defects = sum(len(r["detections"]) for r in results)
        total_time = sum(r["processing_time_s"] for r in results)
        all_confs = [d["confidence"] for r in results for d in r["detections"]]
        mean_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

        m1, m2, m3, m4 = st.columns(4, gap="small")
        with m1:
            st.metric("Images Processed", len(results))
        with m2:
            st.metric("Total Defects Found", total_defects)
        with m3:
            st.metric("Processing Time", f"{total_time:.2f} s")
        with m4:
            st.metric("Mean Confidence", f"{mean_conf:.1%}" if all_confs else "—")

        st.markdown("---")
        st.markdown("**Annotated Results**")

        COLS = 3
        for i in range(0, len(results), COLS):
            batch = results[i : i + COLS]
            cols = st.columns(COLS, gap="small")
            for col, r in zip(cols, batch):
                with col:
                    img = Image.open(r["output_path"])
                    n = len(r["detections"])
                    status = "🔴 DEFECT" if n else "🟢 PASS"
                    classes = ", ".join(sorted({d["class"] for d in r["detections"]})) or "—"
                    st.image(
                        img,
                        caption=f"{r['filename'][:28]} · {status} · {n} · {classes}",
                        use_container_width=True,
                    )

        with st.expander("Raw results JSON"):
            st.json(results)


# ── Tab 2: Conveyor Simulation ─────────────────────────────────────────────────
with tab2:
    st.markdown("#### Conveyor Belt Simulation")
    st.caption(
        "Streams images through the detector one by one, mimicking a real-time conveyor "
        "inspection line. Inference runs live on each frame."
    )

    fps = st.slider(
        "Inspection Speed (images / sec)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5,
        key="conveyor_fps",
    )

    run_conveyor = st.button("▶ Start Conveyor", type="primary", key="run_conveyor")

    if run_conveyor:
        input_dir = _INPUT_DIRS[model_choice]
        image_paths = sorted(
            p for p in input_dir.iterdir() if p.suffix.lower() in _IMAGE_EXT
        )

        if not image_paths:
            st.error("No images found in the input folder.")
        else:
            model = YOLO(str(_MODELS[model_choice]))
            delay = 1.0 / fps
            is_pcb = "pcb" in _MODELS[model_choice].stem.lower()

            header_slot = st.empty()
            img_slot = st.empty()
            status_slot = st.empty()
            prog = st.progress(0)

            pass_count = fail_count = 0

            for idx, img_path in enumerate(image_paths):
                img_bgr = cv2.imread(str(img_path))
                img_for_model = (
                    cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                    if is_pcb else img_bgr
                )

                result = model.predict(
                    img_for_model, conf=conf_threshold, save=False, verbose=False
                )[0]
                annotated = cv2.cvtColor(result.plot(img=img_bgr), cv2.COLOR_BGR2RGB)

                n = len(result.boxes)
                if n:
                    fail_count += 1
                else:
                    pass_count += 1

                header_slot.markdown(
                    f"**[{idx+1}/{len(image_paths)}]** `{img_path.name}` · "
                    f"🟢 PASS: {pass_count} · 🔴 FAIL: {fail_count}"
                )

                if n:
                    classes = ", ".join(
                        sorted({result.names[int(b.cls)] for b in result.boxes})
                    )
                    status_slot.error(
                        f"🔴 DEFECT DETECTED — {n} instance(s) · Classes: {classes}"
                    )
                else:
                    status_slot.success("🟢 PASS — No defects detected")

                img_slot.image(annotated, caption=img_path.name, use_container_width=True)
                prog.progress((idx + 1) / len(image_paths))
                time.sleep(delay)

            header_slot.markdown(
                f"**Simulation complete** · {len(image_paths)} images processed · "
                f"🟢 PASS: {pass_count} · 🔴 FAIL: {fail_count}"
            )
            status_slot.empty()


# ── Tab 3: Inference Pipeline ──────────────────────────────────────────────────
with tab3:
    st.markdown("#### End-to-End Inference Pipeline")
    st.caption(
        "Runs batch inference on the full dataset, then generates a structured PDF audit "
        "report. The report is also available to download directly from this page."
    )

    run_pipeline = st.button("▶ Run Full Pipeline", type="primary", key="run_pipeline")

    if run_pipeline:
        input_dir = _INPUT_DIRS[model_choice]
        model_path = _MODELS[model_choice]

        with st.spinner("Step 1 / 2 — Running batch inference…"):
            results = predict_batch(
                str(model_path), str(input_dir), conf_threshold, str(_OUTPUT_DIR)
            )

        if not results:
            st.error("No images processed. Check the input folder.")
        else:
            st.success(f"Inference complete — {len(results)} images processed.")

            with st.spinner("Step 2 / 2 — Generating PDF report…"):
                report_path = generate_report(
                    results, str(input_dir), str(_REPORTS_DIR), model_path.stem
                )

            st.session_state.pipeline_results = results
            st.session_state.pipeline_report_path = report_path

    if st.session_state.pipeline_results:
        results = st.session_state.pipeline_results
        total_defects = sum(len(r["detections"]) for r in results)
        total_time = sum(r["processing_time_s"] for r in results)
        all_confs = [d["confidence"] for r in results for d in r["detections"]]
        mean_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

        m1, m2, m3, m4 = st.columns(4, gap="small")
        with m1:
            st.metric("Images Processed", len(results))
        with m2:
            st.metric("Total Defects Found", total_defects)
        with m3:
            st.metric("Processing Time", f"{total_time:.2f} s")
        with m4:
            st.metric("Mean Confidence", f"{mean_conf:.1%}" if all_confs else "—")

        if st.session_state.pipeline_report_path:
            report_path = Path(st.session_state.pipeline_report_path)
            st.info(f"PDF report saved → `{report_path.name}`")
            with open(report_path, "rb") as f:
                st.download_button(
                    "📄 Download PDF Report",
                    f.read(),
                    file_name=report_path.name,
                    mime="application/pdf",
                    type="primary",
                )

        with st.expander("Raw results JSON"):
            st.json(results)


st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# OTHER MODULES (Coming Soon)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<span class="badge badge-gray">MODULE 2 — HUMAN-IN-THE-LOOP CORRECTION</span>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="coming-soon-box">'
    "🚧 <strong>Coming Soon</strong> — HITL annotation workflow, Label Studio integration, "
    "and expert review queue."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<span class="badge badge-gray">MODULE 3 — REPORT GENERATION</span>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="coming-soon-box">'
    "🚧 <strong>Coming Soon</strong> — Automated audit reports, detection logs, "
    "bounding box exports, and model performance summaries."
    "</div>",
    unsafe_allow_html=True,
)

st.divider()
st.caption(
    "🔍 Industrial Defect Detection & HITL Pipeline · Final Year Project · "
    "Automated Annotation and Continuous Learning"
)
