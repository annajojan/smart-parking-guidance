# System Workflow

## End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW DIAGRAM                                     │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: INPUT
═══════════════════════════════════════════════════════════════════════════════

  ┌──────────────────┐
  │  CCTV Feed /     │
  │  Upload Image /  │──────► Supported formats:
  │  Upload Video    │       • Images: JPG, JPEG, PNG
  └──────────────────┘       • Videos: MP4, AVI, MOV

                              │
                              ▼
STEP 2: VEHICLE DETECTION
═══════════════════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────────┐
  │                                                                          │
  │   Frame ──► RegionDetector ──► List[Detection]                           │
  │             (edge density)      (bounding boxes)                         │
  │                                                                          │
  │   Each Detection contains:                                               │
  │   • bbox: (x1, y1, x2, y2)                                              │
  │   • label: "car" / "truck"                                               │
  │   • confidence: 0.0 - 1.0                                               │
  │                                                                          │
  └──────────────────────────────────────────────────────────────────────────┘

                              │
                              ▼
STEP 3: OCCUPANCY MATCHING
═══════════════════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────────┐
  │                                                                          │
  │   Detections ──► match_detections_to_slots() ──► OccupancyFrame          │
  │                   (src/occupancy.py)              │                      │
  │                                                    │                      │
  │   For each slot:                                   │                      │
  │   • Calculate intersection area with detections    │                      │
  │   • Check if detection center is inside polygon    │                      │
  │   • If coverage >= 22%: OCCUPIED                   │                      │
  │   • Otherwise: AVAILABLE                           │                      │
  │                                                    ▼                      │
  │                                                                          │
  │   Output: {                                                              │
  │     "total_slots": 8,                                                    │
  │     "available_slots": 3,                                                │
  │     "occupied_slots": 5,                                                 │
  │     "slots": [                                                           │
  │       {"slot_id": "A1", "status": "occupied", "confidence": 0.95},       │
  │       {"slot_id": "A2", "status": "available", "confidence": 1.0},       │
  │       ...                                                                │
  │     ]                                                                    │
  │   }                                                                      │
  │                                                                          │
  └──────────────────────────────────────────────────────────────────────────┘

                              │
                              │  Real occupancy data
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
STEP 4a: LOCAL LLM              STEP 4b: LOCAL IMAGE GENERATION
═══════════════════════════      ══════════════════════════════════════════════

  ┌────────────────────────┐      ┌────────────────────────────────────┐
  │                        │      │                                    │
  │  Occupancy JSON ──►    │      │    Occupancy JSON ──►              │
  │                        │      │                                    │
  │  Ollama Server         │      │    Stable Diffusion Pipeline       │
  │  (localhost:11434)     │      │    (local, CPU/GPU)                │
  │                        │      │                                    │
  │  Model: Llama 3.1      │      │    Model: SD v1.5                  │
  │                        │      │                                    │
  │  System prompt:        │      │    Prompt constructed from:        │
  │  "You are a parking    │      │    • Available slot IDs            │
  │   guidance assistant"  │      │    • Recommended slot              │
  │                        │      │    • Parking lot layout            │
  │  User prompt:          │      │                                    │
  │  Real occupancy data   │      │    Output: PIL Image               │
  │                        │      │    (parking visualization)         │
  │  Output: Natural       │      │                                    │
  │  language directions   │      │                                    │
  │                        │      │                                    │
  └───────────┬────────────┘      └──────────────────┬─────────────────┘
              │                                      │
              ▼                                      ▼
STEP 5: FINAL OUTPUT
═══════════════════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────────┐
  │                                                                          │
  │  1. ANNOTATED IMAGE                                                      │
  │     • Original frame with colored overlays                               │
  │     • Green boxes = Available slots                                      │
  │     • Red boxes = Occupied slots                                         │
  │     • Status bar with counts                                             │
  │                                                                          │
  │  2. OCCUPANCY METRICS                                                    │
  │     • Total slots                                                        │
  │     • Occupied slots                                                     │
  │     • Available slots                                                    │
  │     • Occupancy percentage                                               │
  │                                                                          │
  │  3. AI PARKING GUIDANCE (from Ollama LLM)                                │
  │     • "Parking slot A3 is available. Please drive forward and            │
  │        take the third bay on your left."                                 │
  │                                                                          │
  │  4. AI GENERATED IMAGE (from Stable Diffusion)                           │
  │     • Visual representation of the parking situation                     │
  │     • Recommended slot highlighted                                       │
  │                                                                          │
  └──────────────────────────────────────────────────────────────────────────┘
```

## Streamlit Web Interface Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT WORKFLOW                                   │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────┐
  │                                                                          │
  │   1. Open browser: http://localhost:8501                                 │
  │                                                                          │
  │   2. Sidebar shows:                                                      │
  │      • Project description                                               │
  │      • How the system works                                              │
  │      • AI components (Ollama + Stable Diffusion)                         │
  │      • Input formats                                                     │
  │      • Technology stack                                                  │
  │                                                                          │
  │   3. Main area:                                                          │
  │      • Upload widget (image or video)                                    │
  │      • "Analyze Parking" button                                          │
  │                                                                          │
  │   4. After analysis:                                                     │
  │      • Original image vs annotated image (side by side)                  │
  │      • Occupancy metrics (4 columns)                                     │
  │      • Slot details table                                                │
  │      • Recommended slot (highlighted)                                    │
  │      • AI guidance text (from Ollama)                                    │
  │      • AI generated image (from Stable Diffusion)                        │
  │                                                                          │
  └──────────────────────────────────────────────────────────────────────────┘
```
