"""Smart Parking Guidance System - Streamlit Web Interface.

Run:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import streamlit as st

from src.config import ParkingMap
from src.guidance.rule_based import RuleBasedGuidance
from src.pipeline import ParkingPipeline
from src.visualizer import draw_overlay

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Parking Guidance System",
    page_icon="🅿️",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🅿️ Smart Parking Guidance")
    st.markdown("---")

# ── Helpers ──────────────────────────────────────────────────────────────────
MAP_PATH = ROOT / "config" / "parking_map.json"


@st.cache_resource
def load_pipeline() -> ParkingPipeline:
    return ParkingPipeline(map_path=MAP_PATH, guidance=RuleBasedGuidance())


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def annotate_frame(frame: np.ndarray, pipeline: ParkingPipeline, source: str = "upload"):
    """Run the full pipeline on a single frame and return annotated RGB image + result."""
    result = pipeline.process_frame(frame, source=source, generate_guidance=True)
    annotated = frame.copy()
    draw_overlay(
        annotated,
        result.occupancy,
        pipeline.parking_map,
        guidance_text=result.guidance.text if result.guidance else None,
    )
    return bgr_to_rgb(annotated), result


def render_metrics(occ):
    """Display occupancy metrics in the Streamlit UI."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Slots", occ.total_slots)
    c2.metric("Occupied", occ.occupied_slots)
    c3.metric("Available", occ.available_slots)
    pct = (occ.occupied_slots / occ.total_slots * 100) if occ.total_slots else 0
    c4.metric("Occupancy", f"{pct:.0f}%")


def render_slot_table(occ):
    """Show per-slot details."""
    rows = []
    for s in occ.slots:
        rows.append({
            "Slot": s.slot_id,
            "Zone": s.zone,
            "Status": "🟢 Available" if s.status == "available" else "🔴 Occupied",
            "Confidence": f"{s.confidence:.0%}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ── Main page ────────────────────────────────────────────────────────────────
st.title("🅿️ Smart Parking Guidance System")
st.markdown(
    "A smart parking system that uses computer vision to monitor parking-space "
    "occupancy in real time, identify available spaces, and provide clear "
    "guidance to drivers for efficient parking."
)
st.markdown("---")

pipeline = load_pipeline()

uploaded = st.file_uploader(
    "Upload a CCTV image or parking-lot video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    help="Supported: JPG, JPEG, PNG, MP4, AVI, MOV",
)

if uploaded is None:
    st.info("⬆️ Upload a CCTV image or video to begin analysis.")
    st.stop()

file_bytes = uploaded.read()
suffix = Path(uploaded.name).suffix.lower()
is_image = suffix in (".jpg", ".jpeg", ".png")
is_video = suffix in (".mp4", ".avi", ".mov")

if st.button("🔍 Analyze Parking", type="primary", use_container_width=True):
    try:
        if is_image:
            # ── Image analysis ────────────────────────────────────────────
            st.subheader("📸 CCTV Analysis Results")

            arr = np.frombuffer(file_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                st.error("Could not decode the uploaded image. Please try another file.")
                st.stop()

            with st.spinner("Running vehicle detection model..."):
                annotated_img, result = annotate_frame(frame, pipeline, source=uploaded.name)

            orig_rgb = bgr_to_rgb(frame)

            col_orig, col_annot = st.columns(2)
            with col_orig:
                st.markdown("**Original CCTV Frame**")
                st.image(orig_rgb, use_container_width=True)
            with col_annot:
                st.markdown("**Detected Parking Slots**")
                st.image(annotated_img, use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Parking Occupancy")
            render_metrics(result.occupancy)

            st.markdown("---")
            st.subheader("🚗 Slot Details")
            render_slot_table(result.occupancy)

            # Recommended slot
            available = [s for s in result.occupancy.slots if s.status == "available"]
            if available:
                st.success(f"**Recommended Slot:** {available[0].slot_id} (Zone {available[0].zone})")
            else:
                st.warning("No available parking slots detected.")

            # AI Guidance (Ollama LLM)
            if result.guidance:
                st.markdown("---")
                st.subheader("🧠 Parking Guidance")
                st.info(result.guidance.text)
                st.caption(f"Source: {result.guidance.source}")

        elif is_video:
            # ── Video analysis ────────────────────────────────────────────
            st.subheader("🎬 CCTV Video Analysis")

            tmp_in = Path(tempfile.mktemp(suffix=suffix))
            tmp_in.write_bytes(file_bytes)
            out_dir = Path(tempfile.mkdtemp())

            with st.spinner("Processing CCTV video..."):
                video_result = pipeline.process_video(
                    str(tmp_in), out_dir=str(out_dir), sample_every=3,
                )

            st.markdown("---")
            st.subheader("📊 Final Occupancy Snapshot")
            render_metrics(video_result.final_occupancy)

            st.markdown("---")
            st.subheader("🚗 Slot Details (Final Frame)")
            render_slot_table(video_result.final_occupancy)

            # Recommended slot
            available = [s for s in video_result.final_occupancy.slots if s.status == "available"]
            if available:
                st.success(f"**Recommended Slot:** {available[0].slot_id} (Zone {available[0].zone})")

            # AI Guidance (Ollama LLM)
            final_guidance = pipeline.guidance.generate(
                video_result.final_occupancy, pipeline.parking_map,
            )
            if final_guidance:
                st.markdown("---")
                st.subheader("🧠 Parking Guidance")
                st.info(final_guidance.text)
                st.caption(f"Source: {final_guidance.source}")

            # State-change events
            if video_result.events:
                st.markdown("---")
                st.subheader("📋 State-Change Events")
                event_rows = [
                    {
                        "Frame": e.frame_index,
                        "Slot": e.slot_id,
                        "From": e.previous,
                        "To": e.current,
                    }
                    for e in video_result.events
                ]
                st.dataframe(event_rows, use_container_width=True, hide_index=True)

            # Annotated video download
            ann_path = video_result.files.get("annotated_video")
            if ann_path and Path(ann_path).exists():
                st.markdown("---")
                st.subheader("⬇️ Download Annotated Video")
                with open(ann_path, "rb") as f:
                    st.download_button(
                        "Download Processed Video (MP4)",
                        data=f,
                        file_name=Path(ann_path).name,
                        mime="video/mp4",
                    )

            tmp_in.unlink(missing_ok=True)

        else:
            st.error(f"Unsupported file type: {suffix}")

    except Exception as exc:
        st.error(f"Processing failed: {exc}")
        st.caption("Please ensure the file is a valid parking-lot image or video.")
