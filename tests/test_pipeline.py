import json

from src.guidance.llm import LLMGuidance
from src.guidance.rule_based import RuleBasedGuidance
from src.pipeline import ParkingPipeline
from src.scene_generator import generate_scene, save_scene


def test_end_to_end_synthetic(parking_map, map_path, tmp_path):
    scene = generate_scene(parking_map, seed=3)
    image_path, truth_path = save_scene(scene, tmp_path / "samples", tag="scene")

    pipeline = ParkingPipeline(map_path=map_path)
    result = pipeline.process_image(image_path, out_dir=tmp_path / "outputs")

    assert result.occupancy.total_slots == parking_map.metadata["capacity"]
    assert result.guidance is not None and result.guidance.text

    data = json.loads((tmp_path / "outputs" / "scene_occupancy.json").read_text(encoding="utf-8"))
    assert data["available_slots"] + data["occupied_slots"] == data["total_slots"]

    predicted = {s.slot_id: s.status for s in result.occupancy.slots}
    hits = sum(1 for sid in scene.occupied_ids if predicted.get(sid) == "occupied")
    assert hits / len(scene.occupied_ids) >= 0.75

    assert (tmp_path / "outputs" / "scene_guidance.txt").exists()
    assert (tmp_path / "outputs" / "scene_annotated.png").exists()


def test_llm_fallback_to_rule(parking_map, map_path):
    pipeline = ParkingPipeline(map_path=map_path)
    llm = LLMGuidance(
        api_key="invalid-key",
        base_url="http://127.0.0.1:1",
        timeout=3,
        fallback=RuleBasedGuidance(),
    )

    scene = generate_scene(parking_map, seed=5)
    occupancy = pipeline.process_frame(scene.image, source="test").occupancy
    result = llm.generate(occupancy, pipeline.parking_map)
    assert result.source.startswith("rule")
    assert result.text
