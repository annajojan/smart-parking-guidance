# System Architecture

## Overview

The Smart Parking Guidance System uses computer vision to detect parking occupancy and AI models (running locally) to generate parking guidance and visualizations.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SMART PARKING GUIDANCE SYSTEM                           │
│                        System Architecture                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│   INPUT SOURCE   │
│  CCTV / Video /  │
│  Image Upload    │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPUTER VISION PIPELINE                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Vehicle Detection (src/detectors/)                                 │   │
│  │  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐     │   │
│  │  │ Region       │  │ Saturation       │  │ YOLOv8           │     │   │
│  │  │ Detector     │  │ Detector         │  │ (Optional)       │     │   │
│  │  │ (Edge-based) │  │ (Color-based)    │  │ (Deep Learning)  │     │   │
│  │  └──────────────┘  └──────────────────┘  └──────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Occupancy Detection (src/occupancy.py)                             │   │
│  │  • Match vehicle bounding boxes to parking slot polygons            │   │
│  │  • Calculate intersection-over-slot coverage                        │   │
│  │  • Label each slot: AVAILABLE or OCCUPIED                           │   │
│  │  • Output: OccupancyFrame (structured JSON)                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Visualization (src/visualizer.py)                                  │   │
│  │  • Green rectangles = Available slots                               │   │
│  │  • Red rectangles = Occupied slots                                  │   │
│  │  • Status bar with counts                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         │  Structured Occupancy Data
         │  (slot_id, status, confidence)
         │
         ├──────────────────────────────────┐
         │                                  │
         ▼                                  ▼
┌─────────────────────┐        ┌─────────────────────────┐
│   LOCAL LLM         │        │  LOCAL IMAGE GENERATION  │
│   (Ollama)          │        │  (Stable Diffusion)      │
│                     │        │                          │
│  Model: Llama 3.1   │        │  Model: SD v1.5          │
│  Runs: Local        │        │  Runs: Local             │
│  API: localhost     │        │  No cloud API            │
│  No cloud API       │        │                          │
│                     │        │  Input: Occupancy data   │
│  Input: Real        │        │  Output: Parking viz     │
│  occupancy JSON     │        │  image                   │
│                     │        │                          │
│  Output: Natural    │        │  Prompt constructed      │
│  language parking   │        │  from real slot data     │
│  guidance text      │        │                          │
└─────────┬───────────┘        └───────────┬─────────────┘
          │                                │
          ▼                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FINAL OUTPUT                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐     │
│  │ Annotated Image  │  │ Parking Guidance │  │ Generated Parking    │     │
│  │ (overlay with    │  │ (LLM text with   │  │ Visualization Image  │     │
│  │  slot status)    │  │  directions)     │  │ (AI-generated)       │     │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Vehicle Detection (`src/detectors/`)

| Detector | Method | Requirements | Best For |
|----------|--------|--------------|----------|
| RegionDetector | Per-slot edge density (Canny) | OpenCV only | Aerial/bird's-eye views |
| SaturationDetector | HSV color blob detection | OpenCV only | General parking lots |
| YOLODetector | YOLOv8 deep learning | ultralytics + PyTorch | Real CCTV footage |

### 2. Occupancy Detection (`src/occupancy.py`)

- Matches vehicle bounding boxes to parking slot polygons
- Uses intersection-over-slot-area metric
- Also checks if detection center falls inside slot polygon
- Outputs structured `OccupancyFrame` with per-slot status and confidence

### 3. Local LLM Guidance (`src/guidance/llm.py`)

- Uses Ollama running locally on `http://localhost:11434`
- Model: Llama 3.1 (or any Ollama-compatible model)
- Receives REAL occupancy data from the detection pipeline
- Generates natural-language parking directions
- Falls back to rule-based guidance if Ollama is unavailable

### 4. Local Image Generation (`src/image_generation/`)

- Uses Stable Diffusion via Hugging Face `diffusers` library
- Runs locally on CPU or GPU
- Generates parking visualization based on occupancy data
- No cloud API required

## Data Flow

```
Image/Video → Detection → OccupancyFrame → ┬─→ Ollama LLM → Guidance Text
                                            └─→ Stable Diffusion → Visualization Image
```
