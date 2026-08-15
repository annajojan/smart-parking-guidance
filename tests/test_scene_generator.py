import numpy as np

from src.scene_generator import generate_scene


def test_scene_shape_and_dtype(parking_map):
    scene = generate_scene(parking_map, seed=1)
    assert scene.image.dtype == np.uint8
    assert scene.image.shape[:2] == tuple(reversed(parking_map.image_size))


def test_occupied_ids_valid(parking_map):
    scene = generate_scene(parking_map, seed=7)
    all_ids = {s.slot_id for s in parking_map.slots}
    assert set(scene.occupied_ids) <= all_ids
    assert len(scene.occupied_ids) > 0
    assert set(scene.cars.keys()) == set(scene.occupied_ids)


def test_explicit_occupied_ids(parking_map):
    scene = generate_scene(parking_map, occupied_ids=["A1", "B4"], seed=0)
    assert scene.occupied_ids == ["A1", "B4"]


def test_deterministic_seed(parking_map):
    a = generate_scene(parking_map, seed=99)
    b = generate_scene(parking_map, seed=99)
    assert a.occupied_ids == b.occupied_ids
