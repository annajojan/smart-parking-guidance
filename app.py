"""Smart Parking Guidance System - command-line entry point.

Examples:
    python app.py --mode synthetic --samples 5 --seed 42
    python app.py --mode image --source data/inputs/some_lot.png --detector yolo
    python app.py --mode video --source data/inputs/lot.mp4 --guidance llm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ParkingMap, load_env_file
from src.detectors.region_detector import RegionDetector
from src.detectors.saturation_detector import SaturationDetector
from src.guidance.llm import OllamaGuidance
from src.guidance.rule_based import RuleBasedGuidance
from src.pipeline import ParkingPipeline
from src.scene_generator import generate_scene, save_scene

DEFAULT_MAP = ROOT / "config" / "parking_map.json"
DEFAULT_OUT = ROOT / "data" / "outputs"
DEFAULT_SAMPLES = ROOT / "data" / "samples"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI-powered Smart Parking Guidance System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["synthetic", "image", "video"], default="synthetic", help="input source mode")
    parser.add_argument("--source", default=None, help="path to input image or video")
    parser.add_argument("--map", default=str(DEFAULT_MAP), help="parking map JSON")
    parser.add_argument("--detector", choices=["region", "saturation", "yolo"], default="region", help="vehicle detector backend")
    parser.add_argument("--guidance", choices=["rule", "llm"], default="rule", help="guidance generator")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    parser.add_argument("--samples", type=int, default=3, help="number of synthetic scenes (synthetic mode)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic scenes")
    parser.add_argument("--sample-every", type=int, default=3, help="process every Nth video frame (video mode)")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N video frames")
    return parser


def build_pipeline(args: argparse.Namespace) -> ParkingPipeline:
    load_env_file()
    parking_map = ParkingMap.from_file(args.map)

    if args.detector == "saturation":
        detector = SaturationDetector()
    elif args.detector == "yolo":
        try:
            from src.detectors.yolo_detector import YOLODetector

            detector = YOLODetector()
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            print(f"[warn] YOLO detector unavailable ({exc}); using region detector.", file=sys.stderr)
            detector = RegionDetector(parking_map)
    else:
        detector = RegionDetector(parking_map)

    if args.guidance == "llm":
        guidance = OllamaGuidance(fallback=RuleBasedGuidance())
    else:
        guidance = RuleBasedGuidance()

    return ParkingPipeline(map_path=args.map, detector=detector, guidance=guidance)


def run_synthetic(args: argparse.Namespace, pipeline: ParkingPipeline) -> None:
    parking_map = ParkingMap.from_file(args.map)
    samples_dir = DEFAULT_SAMPLES
    samples_dir.mkdir(parents=True, exist_ok=True)

    for index in range(1, args.samples + 1):
        seed = args.seed + index
        scene = generate_scene(parking_map, seed=seed)
        image_path, truth_path = save_scene(scene, samples_dir, tag=f"scene_{index:02d}")
        print(f"\n=== Synthetic scene {index} ({image_path.name}) ===")
        print(f"Ground truth occupied: {', '.join(scene.occupied_ids)}")

        result = pipeline.process_image(image_path, out_dir=args.out)
        _report(result, show_guidance=True)

        predicted = {s.slot_id: s.status for s in result.occupancy.slots}
        correct = sum(1 for sid in scene.occupied_ids if predicted.get(sid) == "occupied")
        accuracy = correct / max(1, len(scene.occupied_ids))
        print(f"Detection accuracy vs ground truth: {accuracy:.0%} ({correct}/{len(scene.occupied_ids)} occupied slots detected)")


def run_image(args: argparse.Namespace, pipeline: ParkingPipeline) -> None:
    if not args.source:
        raise SystemExit("--mode image requires --source <path>")
    result = pipeline.process_image(args.source, out_dir=args.out)
    _report(result, show_guidance=True)


def run_video(args: argparse.Namespace, pipeline: ParkingPipeline) -> None:
    if not args.source:
        raise SystemExit("--mode video requires --source <path>")
    video = pipeline.process_video(
        args.source,
        out_dir=args.out,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
    )
    print(f"\n=== Video analysis: {args.source} ===")
    print(f"State-change events detected: {len(video.events)}")
    for event in video.events:
        print(f"  frame {event.frame_index}: {event.slot_id} {event.previous} -> {event.current}")
    print(f"Final occupancy: {video.final_occupancy.available_slots}/{video.final_occupancy.total_slots} free")
    for path_key, path_value in video.files.items():
        print(f"  {path_key}: {path_value}")

    final = pipeline.guidance.generate(video.final_occupancy, pipeline.parking_map)
    print(f"\nGuidance [{final.source}]:\n{final.text}\n")


def _report(result, show_guidance: bool) -> None:
    occ = result.occupancy
    print(f"Free: {occ.available_slots}/{occ.total_slots} | Occupied: {occ.occupied_slots}")
    for slot in occ.slots:
        marker = "FREE" if slot.status == "available" else "FULL"
        print(f"  {slot.slot_id:<3} {marker:<5} conf={slot.confidence}")
    if show_guidance and result.guidance is not None:
        print(f"\nGuidance [{result.guidance.source}]:\n{result.guidance.text}\n")
    for key, value in result.files.items():
        if value:
            print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = build_pipeline(args)

    print("Smart Parking Guidance System")
    print(f"Map: {args.map} | Detector: {pipeline.detector.name} | Guidance: {args.guidance}")

    if args.mode == "synthetic":
        run_synthetic(args, pipeline)
    elif args.mode == "image":
        run_image(args, pipeline)
    else:
        run_video(args, pipeline)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
