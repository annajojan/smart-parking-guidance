# Smart Parking Guidance System

An AI-powered smart-parking system that uses **camera-based computer vision** to
detect available and occupied parking slots in real time, converts the detection
results into **structured data**, and passes them to **local AI models** (LLM +
image generation) to generate clear, human-friendly parking directions and
visualizations.

**No cloud AI APIs are required.** All AI inference runs locally on your machine.

---

## Problem Statement

Drivers waste significant time searching for parking spaces, leading to
 congestion, increased emissions, and poor user experience. This system uses
 CCTV cameras and computer vision to identify occupied and available parking
 spaces in real time and provides intelligent guidance to drivers toward
 available parking spaces.

## Motivation

- Reduce time drivers spend searching for parking
- Cut traffic congestion and emissions in parking facilities
- Improve overall parking experience
- Demonstrate local AI inference without cloud dependencies

## Objectives

1. Detect vehicle presence in parking slots using computer vision
2. Classify each slot as available or occupied
3. Generate natural-language parking guidance using a local LLM
4. Generate parking visualization images using a local image generation model
5. Provide a clean web interface for demonstration

## Features

- Real-time slot occupancy detection from images, videos, or generated scenes
- Three detection backends:
  - `region` – per-slot interior edge-density analysis (default, offline)
  - `saturation` – colour-based foreground segmentation
  - `yolo` – Ultralytics YOLOv8 for highest accuracy (optional)
- Structured, JSON-serialisable occupancy output with confidence scores
- **Local LLM guidance** via Ollama (Llama 3.1) – no cloud API required
- **Local image generation** via Stable Diffusion – no cloud API required
- Annotated output images/videos with color-coded slots
- Synthetic scene generator for demos, tests, and CI
- Video analysis with slot state-change event tracking
- Streamlit web interface for easy demonstration

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SMART PARKING GUIDANCE SYSTEM                     │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  CCTV /      │
│  Image /     │
│  Video       │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. Vehicle Detection (src/detectors/)                               │
│     • RegionDetector (edge density)                                  │
│     • SaturationDetector (color blobs)                               │
│     • YOLOv8 (deep learning, optional)                               │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ bounding boxes
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. Occupancy Detection (src/occupancy.py)                          │
│     • Match detections to slot polygons                              │
│     • Label: AVAILABLE / OCCUPIED                                    │
│     • Output: OccupancyFrame (structured JSON)                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ structured data
                            ├────────────────────────┐
                            │                        │
                            ▼                        ▼
┌──────────────────────┐  ┌──────────────────────────────────────┐
│  3a. Local LLM       │  │  3b. Local Image Generation          │
│  (Ollama + Llama 3.1)│  │  (Stable Diffusion)                  │
│                      │  │                                      │
│  Input: Real         │  │  Input: Occupancy data               │
│  occupancy JSON      │  │  Output: Parking visualization       │
│  Output: Natural     │  │  image                               │
│  language guidance   │  │                                      │
└──────────┬───────────┘  └──────────────────┬───────────────────┘
           │                                 │
           ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. Final Output                                                     │
│     • Annotated image with slot status                               │
│     • AI-generated parking guidance text                             │
│     • AI-generated parking visualization image                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Workflow

```
CCTV Feed / Image Upload / Video Upload
        │
        ▼
Vehicle Detection (RegionDetector / YOLOv8)
        │
        ▼
Occupancy Matching (per-slot classification)
        │
        ├────────────────────────┐
        │                        │
        ▼                        ▼
   Local LLM               Local Image Generation
   (Ollama)                (Stable Diffusion)
        │                        │
        ▼                        ▼
   Text Guidance            Visualization Image
        │                        │
        └────────┬───────────────┘
                 │
                 ▼
        Final Smart Parking Result
```

## Technologies

| Component | Technology | Description |
|-----------|-----------|-------------|
| Vehicle Detection | OpenCV + RegionDetector | Edge-density per-slot analysis |
| Vehicle Detection (alt) | YOLOv8 | Deep learning vehicle detection |
| Occupancy Detection | Custom geometry matching | Intersection-over-slot + polygon test |
| Text Generation | Ollama + Llama 3.1 | Local LLM, no cloud API |
| Image Generation | Stable Diffusion v1.5 | Local image generation, no cloud API |
| Web Interface | Streamlit | Interactive web demo |
| Language | Python 3.10+ | Core implementation |

## Project Structure

```
smart-parking-guidance/
├── app.py                        # CLI entry point
├── streamlit_app.py              # Web interface
├── config/
│   └── parking_map.json          # Slot polygons, zones, entrances
├── src/
│   ├── config.py                 # Parking map loading
│   ├── scene_generator.py        # Synthetic parking-lot renderer
│   ├── detectors/                # CV detectors
│   │   ├── region_detector.py    # Edge-density detector
│   │   ├── saturation_detector.py# Color-based detector
│   │   └── yolo_detector.py      # YOLOv8 detector
│   ├── occupancy.py              # Structured data + slot matcher
│   ├── guidance/                 # Guidance generators
│   │   ├── rule_based.py         # Deterministic directions
│   │   └── llm.py                # Ollama local LLM
│   ├── image_generation/         # Image generation
│   │   └── stable_diffusion.py   # Stable Diffusion generator
│   ├── visualizer.py             # Annotated overlay rendering
│   └── pipeline.py               # End-to-end pipeline
├── models/
│   └── README.md                 # Model download instructions
├── data/
│   ├── inputs/                   # Your images/videos
│   ├── outputs/                  # Pipeline results
│   └── samples/                  # Generated synthetic scenes
├── docs/
│   ├── architecture.md           # System architecture
│   ├── workflow.md               # Workflow diagrams
│   └── screenshots/              # Application screenshots
├── demo/
│   └── demo.mp4                  # Demo video
└── tests/                        # Test suite
```

## Installation

### Prerequisites

- Python 3.10+ (tested on 3.12)
- Ollama (for local LLM)
- ~9 GB disk space for AI models

### Step 1: Clone and set up Python environment

```bash
git clone https://github.com/annajojan/smart-parking-guidance.git
cd smart-parking-guidance

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Install Ollama (Local LLM)

1. Download from https://ollama.com/download
2. Install and start the server:
   ```bash
   ollama serve
   ```
3. Pull the Llama 3.1 model:
   ```bash
   ollama pull llama3.1
   ```

### Step 3: Download Stable Diffusion (Local Image Generation)

The model downloads automatically on first use (~4 GB). To pre-download:

```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5")
```

See `models/README.md` for detailed instructions.

## Usage

### CLI

```bash
# Generate synthetic scenes and detect occupancy
python app.py --mode synthetic --samples 3 --seed 42

# Process an image with Ollama LLM guidance
python app.py --mode image --source data/inputs/my_lot.png --guidance llm

# Analyse a video
python app.py --mode video --source data/inputs/my_lot.mp4

# Use YOLO detector (requires ultralytics)
python app.py --mode image --source data/inputs/my_lot.png --detector yolo
```

### Web Interface (Streamlit)

```bash
streamlit run streamlit_app.py
```

Opens browser at `http://localhost:8501`.

### Demo Steps

1. Open the web interface
2. Upload a parking-lot image (try `data/samples/scene_01.png`)
3. Click "Analyze Parking"
4. View results:
   - Original vs annotated image
   - Occupancy metrics
   - Slot details table
   - AI guidance text (from Ollama)
   - AI generated visualization (from Stable Diffusion)

## Screenshots

See `docs/screenshots/` for application screenshots.

## Demo Video

See `demo/demo.mp4` for a recorded demonstration.

## Configuration

### Parking Map

Edit `config/parking_map.json` to define your parking lot:
- Image size
- Slot polygons (bird's-eye coordinates)
- Zones and levels
- Entrances
- Drive lane boundaries

### Environment Variables

Create a `.env` file (never commit this):

```bash
# Ollama LLM configuration
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.1
```

## Tests

```bash
pytest -v
```

## Limitations

- Requires sufficient disk space for AI models (~9 GB)
- Stable Diffusion runs slowly on CPU (~30-120 seconds per image)
- YOLOv8 requires PyTorch and a GPU for real-time performance
- Parking slot polygons must be defined manually for each lot
- Synthetic scenes are simplified representations of real parking lots

## Future Improvements

- Real-time CCTV video stream processing
- Mobile app interface
- Multi-floor parking support
- Integration with parking payment systems
- Real-time occupancy tracking dashboard
- Support for more image generation models (FLUX, SDXL)

## License

[MIT](LICENSE)
