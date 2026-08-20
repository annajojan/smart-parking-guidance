# Smart Parking Guidance System

An AI-powered smart-parking system that uses **camera-based computer vision** to
detect available and occupied parking slots in real time, converts the detection
results into **structured data**, and passes them to a **local LLM** to generate
clear, standardized parking guidance.

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

## Objectives

1. Detect vehicle presence in parking slots using computer vision
2. Classify each slot as available or occupied
3. Generate standardized parking guidance using a local LLM
4. Provide a clean web interface for demonstration

## Features

- Real-time slot occupancy detection from images, videos, or generated scenes
- Three detection backends:
  - `region` – per-slot interior edge-density analysis (default, offline)
  - `saturation` – colour-based foreground segmentation
  - `yolo` – Ultralytics YOLOv8 for highest accuracy (optional)
- Structured, JSON-serialisable occupancy output with confidence scores
- **Local LLM guidance** via Ollama (Llama 3.1)
- Standardized SMARTPARK parking guidance template
- Annotated output images/videos with color-coded slots

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
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. Local LLM (Ollama + Llama 3.1)                                  │
│     Input: Real occupancy JSON                                       │
│     Output: Standardized parking guidance text                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. Final Output                                                     │
│     • Annotated image with slot status                               │
│     • Standardized parking guidance text                             │
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
        ▼
Local LLM (Ollama)
        │
        ▼
Standardized Parking Guidance
```

## Technologies

| Component | Technology | Description |
|-----------|-----------|-------------|
| Vehicle Detection | OpenCV + RegionDetector | Edge-density per-slot analysis |
| Vehicle Detection (alt) | YOLOv8 | Deep learning vehicle detection |
| Occupancy Detection | Custom geometry matching | Intersection-over-slot + polygon test |
| Text Generation | Ollama + Llama 3.1 | Local LLM |
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
│   ├── visualizer.py             # Annotated overlay rendering
│   └── pipeline.py               # End-to-end pipeline
├── data/
│   ├── inputs/                   # Your images/videos
│   ├── outputs/                  # Pipeline results
│   └── samples/                  # Generated synthetic scenes
├── docs/
│   ├── architecture.md           # System architecture
│   └── workflow.md               # Workflow diagrams
└── tests/                        # Test suite
```

## Installation

### Prerequisites

- Python 3.10+ (tested on 3.12)
- Ollama (for local LLM)

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
   - Parking guidance text

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

- YOLOv8 requires PyTorch and a GPU for real-time performance
- Parking slot polygons must be defined manually for each lot
- Synthetic scenes are simplified representations of real parking lots

## Future Improvements

- Real-time CCTV video stream processing
- Mobile app interface
- Multi-floor parking support
- Integration with parking payment systems
- Real-time occupancy tracking dashboard

## License

[MIT](LICENSE)
