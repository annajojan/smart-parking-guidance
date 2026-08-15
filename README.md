# Smart Parking Guidance System

An AI-powered smart-parking system that uses **camera-based computer vision** to
detect available and occupied parking slots in real time, converts the detection
results into **structured data**, and passes them to an **LLM (or a deterministic
fallback)** to generate clear, human-friendly parking directions.

The goal: reduce the time drivers spend searching for a spot, cut congestion and
emissions, and improve the overall parking experience.

---

## How it works

```
                 ┌──────────────────────────────────────────────┐
 Camera frame ──▶│ 1. Vehicle detection (computer vision)       │
                 │    - RegionDetector (per-slot, offline)      │
                 │    - SaturationDetector (colour blobs)       │
                 │    - YOLOv8 (optional, higher accuracy)      │
                 └─────────────────────┬────────────────────────┘
                                       │ bounding boxes
                                       ▼
                 ┌──────────────────────────────────────────────┐
                 │ 2. Slot occupancy matching (geometry)        │
                 │    boxes are matched to slot polygons and    │
                 │    each slot is labelled available/occupied  │
                 │    with a confidence score                   │
                 └─────────────────────┬────────────────────────┘
                                       │ structured JSON
                                       ▼
                 ┌──────────────────────────────────────────────┐
                 │ 3. Guidance generation (LLM)                 │
                 │    LLM prompt built from the structured data │
                 │    -> "Park in A3, second bay on your left"  │
                 │    Rule-based fallback works fully offline   │
                 └─────────────────────┬────────────────────────┘
                                       │
                                       ▼
                    Annotated image + occupancy.json + guidance.txt
```

**Pipeline stages** (see `src/pipeline.py`):

1. **Computer vision** – `src/detectors/` turns each camera frame into a list of
   vehicle bounding boxes (`Detection`).
2. **Structured data** – `src/occupancy.py` matches boxes against the slot
   polygons defined in `config/parking_map.json` and produces a structured
   `OccupancyFrame` (available/occupied per slot + confidence).
3. **LLM guidance** – `src/guidance/` builds a prompt from the structured data and
   asks an OpenAI-compatible LLM for friendly step-by-step directions.
   `RuleBasedGuidance` is the deterministic offline fallback.

## Features

- Real-time slot occupancy detection from images, videos or generated scenes
- Three detection backends:
  - `region` – per-slot interior edge-density analysis (default, offline, standard for aerial parking views)
  - `saturation` – colour-based foreground segmentation into car boxes
  - `yolo` – Ultralytics YOLOv8 for highest accuracy on real footage (optional)
- Structured, JSON-serialisable occupancy output with confidence scores
- LLM-generated parking directions (OpenAI-compatible API) with an automatic
  rule-based fallback
- Annotated output images/videos with color-coded slots
- Synthetic scene generator for demos, tests and CI
- Video analysis with slot state-change (free ⇄ occupied) event tracking
- Full test suite + GitHub Actions CI

## Project structure

```
.
├── app.py                        # CLI entry point
├── config/
│   └── parking_map.json          # slot polygons, zones, entrances
├── data/
│   ├── inputs/                   # your own images/videos go here
│   ├── outputs/                  # results (occupancy, guidance, overlays)
│   └── samples/                  # generated synthetic scenes
├── src/
│   ├── config.py                 # parking map loading
│   ├── scene_generator.py        # synthetic parking-lot renderer
│   ├── detectors/                # CV detectors (region, saturation, yolo)
│   ├── occupancy.py              # structured data model + slot matcher
│   ├── guidance/                 # rule-based + LLM generators
│   ├── visualizer.py             # annotated overlay rendering
│   └── pipeline.py               # end-to-end pipeline
└── tests/                        # pytest suite
```

## Getting started

**Requirements:** Python 3.10+ (tested on 3.12).

```bash
# 1. Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) dev dependencies for running the tests
pip install -r requirements-dev.txt
```

## Usage

```bash
# Generate 3 synthetic scenes, detect occupancy, and print guidance
python app.py --mode synthetic --samples 3 --seed 42

# Process one of your own images
python app.py --mode image --source data/inputs/my_lot.png

# Process an image with the YOLO detector (requires requirements-optional.txt)
python app.py --mode image --source data/inputs/my_lot.png --detector yolo

# Analyse a video and track slot state changes
python app.py --mode video --source data/inputs/my_lot.mp4

# Use the LLM for guidance (requires an API key, see below)
python app.py --mode synthetic --guidance llm
```

### Example output

```
=== Synthetic scene 1 (scene_01.png) ===
Ground truth occupied: A1, A2, B1, B3, B4
Free: 3/8 | Occupied: 5/8
  A1  FULL conf=0.93
  A2  FULL conf=0.96
  A3  FREE conf=1.0
  A4  FREE conf=1.0
  B1  FULL conf=0.94
  B2  FREE conf=1.0
  B3  FULL conf=0.95
  B4  FULL conf=0.91

Guidance [rule]:
Good news — 3 of 8 spots are currently free. The nearest free spot is A3 in
Zone A, on your left, about 7 metres ahead. Drive down the aisle and take the
third bay on your left. Pull in carefully — the spot is waiting for you.

  occupancy_json:   data/outputs/scene_01_occupancy.json
  guidance_txt:     data/outputs/scene_01_guidance.txt
  annotated_image:  data/outputs/scene_01_annotated.png
```

Files written to `data/outputs/`:

| File | Contents |
|---|---|
| `*_occupancy.json` | Structured occupancy data (per-slot status + confidence) |
| `*_guidance.txt` | Human-friendly parking directions + source |
| `*_annotated.png` | Frame overlay: green = free, red = occupied |
| `*_summary.json` | Video mode: slot state-change events |

## Using the LLM

The LLM generator calls any OpenAI-compatible `/chat/completions` endpoint
(OpenAI, Azure, Groq, Ollama, llama.cpp, ...). Configuration is read from the
environment or a `.env` file:

```bash
# .env (never commit this file)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=https://api.openai.com/v1   # default
```

If the LLM call fails (no key, offline, timeout), the system **automatically
falls back** to the rule-based generator and reports it in the guidance source,
e.g. `rule (LLM unavailable: URLError)`.

## Extending to your own parking lot

1. Create a map JSON like `config/parking_map.json`: set the image size, lot
   bounds, entrances and the polygon of every parking slot.
2. Point the pipeline at it with `--map path/to/your_map.json`.
3. Use the `yolo` detector for best results on real camera footage, or tune the
   `RegionDetector` / `SaturationDetector` parameters (see
   `src/detectors/`) for your camera angle.

## Tests

```bash
pytest -v
```

The suite covers the detection→slot matcher, the rule-based guidance, the scene
generator, the full end-to-end pipeline, and the LLM→rule fallback. CI runs the
tests plus a demo and uploads the generated artifacts.

## License

[MIT](LICENSE)
