# Vision-Based Navigation Assistant for Low-Vision Users

A computer vision system that provides **actionable navigation guidance** to visually impaired users by analyzing street scenes in real-time.

The system combines **object detection**, **monocular depth estimation**, **temporal reasoning**, and **rule-based navigation planning** to generate natural language instructions such as:

> "Car near ahead. 2 motorcycles on your left. Stay on the left edge."

**Focus:** Cluttered Indian street environments with unique objects like autorickshaws, riders, motorcycles, and animals.

---

## Project Objective

Transform street scene perception into **actionable movement commands**—not just scene descriptions.

Core Questions Answered:

- What hazards are ahead?
- How far are they?
- In which direction?
- What should I do? (STOP / AVOID / MOVE / STAY ON EDGE / CONTINUE FORWARD)

---

## Complete Pipeline Architecture

```
INPUT: Video Frame / Image (640×384)
    │
    ├─→ [1] Object Detection (YOLOv8-IDD)
    │       ├─→ Detects: vehicles, pedestrians, animals
    │       └─→ Output: Bounding boxes + confidence
    │
    ├─→ [2] Depth Estimation (MiDaS)
    │       ├─→ Input: Single RGB image
    │       └─→ Output: Normalized depth [0, 1]
    │
    ├─→ [3] Spatial Reasoning
    │       ├─→ Direction: left/center/right
    │       ├─→ FOV-aware distance: very close → near → moderate → far
    │       ├─→ Risk scoring (depth + size + motion)
    │       └─→ Motion classification: approaching/crossing/receding
    │
    ├─→ [4] Temporal Reasoning (Multi-Frame)
    │       ├─→ Object tracking (DeepSORT)
    │       ├─→ Motion vectors & time-to-collision
    │       └─→ Exponential decay of stale hazards
    │
    ├─→ [5] Scene Memory & Navigation Planning
    │       ├─→ Cost map (zone-based hazard costs)
    │       ├─→ Occupancy grid (spatial probability)
    │       ├─→ Corridor detection (safe routes)
    │       └─→ 8-layer rule-based planner (STOP→AVOID→MOVE→EDGE→FORWARD)
    │
    └─→ [6] Caption Generation, VLM Refinement & Text-to-Speech
        ├─→ Hazard grouping (merge similar objects)
        ├─→ Temporal smoothing (3-frame anti-flicker)
        ├─→ VLM check/refinement for image mode (BLIP VQA)
        ├─→ Multi-line text wrapping
        └─→ Smart TTS gating (urgent vs. passive)

OUTPUT: Audio Guidance + Annotated Video
```

---

## Navigation Rules (Priority Order)

```
Rule 1: Time-to-collision < 1s       → STOP (critical)
Rule 2: Very close hazards           → AVOID (warning)
Rule 3: Center blocked + close       → MOVE LEFT/RIGHT (warning)
Rule 4: Approaching motion           → AVOID (warning)
Rule 5: Both sides blocked           → SUGGEST (info)
Rule 5.7: Road penalty (center > 1.2) → EDGE LEFT/RIGHT (info)
Rule 5.5: Distant hazards only       → CONTINUE FORWARD (info)
Rule 6: Corridor-based routing       → FOLLOW CORRIDOR (info)
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- macOS, Linux, or Windows
- ~2 GB disk space for models

### Quick Start

**1. Clone Repository**

```bash
git clone https://github.com/neutron-69/Team21_low_vision_navigation.git
cd Team21_low_vision_navigation/03_code
```

**2. Clone MiDaS Repository (required)**

```bash
git clone https://github.com/isl-org/MiDaS.git
```

**3. Create Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**4. Install Dependencies**

```bash
pip install -r requirements.txt
```

**5. Download Model Weights**

YOLOv8-IDD weights:

```bash
mkdir -p src/models/weights
# Download: https://github.com/Udit21Ag/vlm-navigation-assistant/releases/download/v1.0/idd_best.pt
# Place in: src/models/weights/idd_best.pt
```

MiDAS depth model:

```bash
mkdir -p MiDAS/weights
# Download: https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt
# Place in: MiDAS/weights/dpt_levit_224.pt
```

**6. Download Sample Files**

Download the sample media to run the examples:

```bash
mkdir -p ../04_data/sample_inputs
# Download sample images and videos via curl
curl -L -o ../04_data/sample_inputs/road2.png https://github.com/neutron-69/Team21_low_vision_navigation/releases/download/samples/road2.png
curl -L -o ../04_data/sample_inputs/sample1.mp4 https://github.com/neutron-69/Team21_low_vision_navigation/releases/download/samples/sample1.mp4
curl -L -o ../04_data/sample_inputs/sample2.mp4 https://github.com/neutron-69/Team21_low_vision_navigation/releases/download/samples/sample2.mp4
curl -L -o ../04_data/sample_inputs/sample_2.mp4 https://github.com/neutron-69/Team21_low_vision_navigation/releases/download/samples/sample_2.mp4
```

---

## Usage

### Image Mode

Analyze a single street image:

```bash
python scripts/infer.py --image ../04_data/sample_inputs/road.jpg
python scripts/infer.py --image ../04_data/sample_inputs/road2.png --no-tts
```

**Output:**

- Terminal: Guidance text + TTS audio
- File: `../05_results/figures/outputs/road_*.jpg` (annotated image)

Image mode also runs VLM-based validation/refinement on the generated caption to improve consistency between detected hazards and final text output.

---

### Video Mode

Process video streams:

```bash
# Standard processing
python scripts/infer.py --source ../04_data/sample_inputs/sample1.mp4

# Without audio
python scripts/infer.py --source ../04_data/sample_inputs/sample2.mp4 --no-tts

# Custom sampling interval (milliseconds)
python scripts/infer.py --source ../04_data/sample_inputs/sample_2.mp4 --interval 300
```

**Features:**

- Adaptive frame sampling (minimum 25 frames for short videos)
- Asynchronous depth & detection (parallel threading)
- Multi-line caption wrapping
- FPS & latency metrics overlay

**Output:**

- Video: `../05_results/figures/outputs/output.mp4`
- Frames: `../05_results/figures/outputs/frame_*.jpg` (every 5 sampled frames)

---

### Webcam (Real-Time)

```bash
python scripts/demo.py
```

---

## Project Structure

```
.
├── main.py                          # Entry point (image + video modes)
├── requirements.txt                 # Dependencies
├── README.md                        # This file
│
├── models/                          # Core ML modules
│   ├── detector.py                  # YOLOv8 object detection
│   ├── depth_estimator.py           # MiDaS monocular depth
│   ├── spatial_reasoning.py         # Direction + distance classification
│   ├── tracker.py                   # DeepSORT object tracking
│   ├── temporal_reasoner.py         # Motion analysis (frame-to-frame)
│   ├── scene_memory.py              # Cost map + corridor estimation
│   ├── navigation_planner.py        # 8-layer rule-based planner
│   ├── frame_sampler.py             # Adaptive video sampling
│   ├── metrics.py                   # Runtime statistics
│   ├── vlm_reasoner.py              # BLIP VQA validation
│   ├── occupancy_grid.py            # Spatial probability grid
│   ├── corridor_estimator.py        # Safe navigation path detection
│   └── weights/
│       └── idd_best.pt              # YOLOv8 fine-tuned on Indian Driving Dataset
│
├── caption/
│   └── temporal_caption.py          # Hazard grouping + smoothing
│
├── tts/
│   └── event_speaker.py             # Non-blocking text-to-speech
│
├── utils/
│   └── visualize.py                 # Bounding box & metric overlay
│
├── MiDAS/                           # Depth estimation (submodule)
│   ├── midas/
│   │   ├── base_model.py
│   │   ├── dpt_depth.py
│   │   ├── model_loader.py
│   │   └── backbones/
│   └── weights/
│       └── dpt_levit_224.pt
│
├── samples/                         # Example inputs
│   ├── road2.png
│   ├── sample1.mp4
│   ├── sample2.mp4
│   └── sample_2.mp4
│
└── outputs/                         # Generated results
    ├── output.mp4
    ├── frame_*.jpg
    └── road_*.jpg
```

---

## Novelty

### 1. Vision-to-Navigation Pipeline

Outputs **actionable commands** (STOP, AVOID, MOVE, EDGE, FORWARD)—not generic scene captions.

### 2. FOV-Aware Distance Classification

Adapts depth thresholds based on object vertical position and size, reflecting camera field-of-view geometry.

### 3. Motion-Aware Navigation

Detects approaching vehicles, crossing pedestrians for better hazard prioritization.

### 4. Road Safety Bias

Center navigation costs more; edges are safer—reflects real-world urban safety.

### 5. Indian Street Support

Includes autorickshaws, riders, motorcycles, animals—classes missing from COCO.

### 6. Real-Time Performance

- Single image: ~315ms (CPU)
- Video: 4-8 FPS with async processing
- Adaptive sampling for any video length

### 7. Temporal Stability

- 3-frame instruction smoothing (anti-flicker)
- Exponential decay for stale hazards
- Smart TTS gating (urgent instructions priority)

---

## Model Information

| Component     | Model                 | Size   | Input       | Output                |
| ------------- | --------------------- | ------ | ----------- | --------------------- |
| **Detection** | YOLOv8n (IDD)         | 6.3 MB | 640×384 RGB | Bboxes + class + conf |
| **Depth**     | MiDaS DPT-LeViT       | 51 MB  | 224×224 RGB | Depth [0, 1]          |
| **Tracking**  | DeepSORT              | <1 MB  | Detections  | Track IDs + motion    |
| **VQA**       | BLIP                  | 900 MB | RGB + text  | Yes/No answers        |
| **Speech**    | macOS `say` / pyttsx3 | System | Text        | Audio                 |
