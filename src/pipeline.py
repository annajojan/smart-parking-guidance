"""End-to-end pipeline: frame -> detections -> structured data -> guidance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2

from .config import ParkingMap, load_env_file
from .detectors.base import BaseDetector
from .detectors.region_detector import RegionDetector
from .guidance.base import BaseGuidance, GuidanceResult
from .guidance.rule_based import RuleBasedGuidance
from .occupancy import OccupancyFrame, match_detections_to_slots
from .visualizer import draw_overlay


@dataclass
class PipelineResult:
    """Everything produced from processing one frame."""

    occupancy: OccupancyFrame
    guidance: Optional[GuidanceResult] = None
    files: dict = field(default_factory=dict)


@dataclass
class SlotEvent:
    """A state transition observed for a slot."""

    slot_id: str
    previous: str
    current: str
    frame_index: int
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "slot_id": self.slot_id,
            "previous": self.previous,
            "current": self.current,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
        }


@dataclass
class VideoResult:
    """Summary of processing a whole video."""

    events: List[SlotEvent]
    final_occupancy: OccupancyFrame
    files: dict = field(default_factory=dict)


class ParkingPipeline:
    """Coordinates the computer-vision and guidance components."""

    def __init__(
        self,
        map_path: str | Path,
        detector: Optional[BaseDetector] = None,
        guidance: Optional[BaseGuidance] = None,
    ) -> None:
        load_env_file()
        self.parking_map = ParkingMap.from_file(map_path)
        self.detector = detector or RegionDetector(self.parking_map)
        self.guidance = guidance or RuleBasedGuidance()

    def process_frame(
        self,
        frame,
        source: str = "frame",
        generate_guidance: bool = True,
        timestamp: Optional[str] = None,
    ) -> PipelineResult:
        detections = self.detector.detect(frame)
        occupancy = match_detections_to_slots(
            self.parking_map,
            detections,
            source=source,
            timestamp=timestamp,
        )
        result = PipelineResult(occupancy=occupancy)
        if generate_guidance:
            result.guidance = self.guidance.generate(occupancy, self.parking_map)
        return result

    def process_image(self, image_path: str | Path, out_dir: str | Path, overlay: bool = True) -> PipelineResult:
        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        result = self.process_frame(frame, source=image_path.name)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = image_path.stem
        occupancy_path = out_dir / f"{stem}_occupancy.json"
        occupancy_path.write_text(result.occupancy.to_json(), encoding="utf-8")

        if result.guidance is not None:
            guidance_path = out_dir / f"{stem}_guidance.txt"
            guidance_path.write_text(
                f"[source: {result.guidance.source}]\n{result.guidance.text}\n",
                encoding="utf-8",
            )
        else:
            guidance_path = None

        if overlay:
            annotated = frame.copy()
            draw_overlay(
                annotated,
                result.occupancy,
                self.parking_map,
                guidance_text=result.guidance.text if result.guidance else None,
            )
            image_out = out_dir / f"{stem}_annotated.png"
            cv2.imwrite(str(image_out), annotated)
        else:
            image_out = None

        result.files = {
            "occupancy_json": str(occupancy_path),
            "guidance_txt": str(guidance_path) if guidance_path else None,
            "annotated_image": str(image_out) if image_out else None,
        }
        return result

    def process_video(
        self,
        video_path: str | Path,
        out_dir: str | Path,
        sample_every: int = 3,
        max_frames: Optional[int] = None,
    ) -> VideoResult:
        video_path = Path(video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        annotated_out = out_dir / f"{video_path.stem}_annotated.mp4"
        writer = cv2.VideoWriter(
            str(annotated_out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        events: List[SlotEvent] = []
        previous: Optional[dict] = None
        final_occupancy: Optional[OccupancyFrame] = None
        frame_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frames is not None and frame_index >= max_frames:
                break

            if frame_index % sample_every == 0:
                result = self.process_frame(frame, source=video_path.name, generate_guidance=False)
                final_occupancy = result.occupancy

                states = {s.slot_id: s.status for s in result.occupancy.slots}
                if previous is not None:
                    for slot_id, status in states.items():
                        old = previous.get(slot_id)
                        if old is not None and old != status:
                            events.append(
                                SlotEvent(
                                    slot_id=slot_id,
                                    previous=old,
                                    current=status,
                                    frame_index=frame_index,
                                    timestamp=result.occupancy.timestamp,
                                )
                            )
                previous = states

            writer.write(frame)
            frame_index += 1

        capture.release()
        writer.release()

        if final_occupancy is None:
            raise ValueError(f"No frames processed from video: {video_path}")

        summary = {
            "events": [e.to_dict() for e in events],
            "total_events": len(events),
            "final_occupancy": final_occupancy.to_dict(),
        }
        summary_path = out_dir / f"{video_path.stem}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return VideoResult(
            events=events,
            final_occupancy=final_occupancy,
            files={
                "annotated_video": str(annotated_out),
                "summary_json": str(summary_path),
            },
        )
