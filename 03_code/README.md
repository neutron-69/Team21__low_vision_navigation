# Vision-Based Navigation Assistant for Low-Vision Users

A computer vision system that provides **actionable navigation guidance** to visually impaired users by analyzing street scenes in real-time.

The system combines **object detection (YOLOv8-IDD)**, **monocular depth estimation (MiDaS)**, **temporal reasoning**, and **rule-based navigation planning** to generate natural language instructions such as:

> "Car near ahead. 2 motorcycles on your left. Stay on the left edge."

**Focus:** Cluttered Indian street environments with unique objects like autorickshaws, riders, motorcycles, and animals.

---

## Environment Setup & System Requirements

### Hardware Used (Development & Testing)

- **OS:** macOS (Apple Silicon / Intel x86)
- **GPU:** Optional (cuda preferred for speed, falls back to CPU)
- **Python:** 3.8+
- **Disk Space:** ~2 GB for models + data
- **RAM:** 4 GB minimum (8 GB recommended for video processing)

### Dependencies

All dependencies are specified in `requirements.txt` and pinned to known-working versions.

**Key packages:**

- `torch==2.10.0` + `torchvision==0.25.0` (PyTorch with CUDA support)
- `ultralytics==8.4.19` (YOLOv8 detector)
- `opencv-python==4.13.0.92` (Computer vision)
- `transformers==5.5.4` (BLIP VLM for image-mode refinement)
- `pyttsx3==2.99` (Text-to-speech)
- `deep-sort-realtime==1.3.2` (Object tracking)

For a complete list, see `requirements.txt`.

---

## Quick Start

### 1. Setup Environment

```bash
cd 03_code

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model Weights

```bash
# YOLOv8-IDD weights (fine-tuned on Indian Driving Dataset)
mkdir -p src/models/weights
curl -L -o src/models/weights/idd_best.pt \
    https://github.com/Udit21Ag/vlm-navigation-assistant/releases/download/v1.0/idd_best.pt

# Clone MiDAS repository (required for depth estimation)
git clone https://github.com/isl-org/MiDaS.git

# MiDAS depth estimation model
mkdir -p MiDAS/weights
curl -L -o MiDAS/weights/dpt_levit_224.pt \
    https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt
```

### 3. Download Sample Data

```bash
mkdir -p ../04_data/sample_inputs
curl -L -o ../04_data/sample_inputs/road.jpg \
    https://github.com/neutron-69/Team21_low_vision_navigation/releases/download/samples/road.jpg
curl -L -o ../04_data/sample_inputs/road2.png \
    https://github.com/neutron-69/Team21_low_vision_navigation/releases/download/samples/road2.png
curl -L -o ../04_data/sample_inputs/sample1.mp4 \
    https://github.com/neutron-69/Team21_low_vision_navigation/releases/download/samples/sample2.mp4
```

---

## Usage

### Inference Commands

All inference runs are executed from the `scripts/` directory or via `src/main.py`.

#### Image Mode

Process a single street image and output annotated result with audio guidance:

```bash
# Using inference script
python scripts/infer.py --image ../04_data/sample_inputs/road.jpg

# Direct main.py
python src/main.py --image ../04_data/sample_inputs/road.jpg

# Without text-to-speech
python scripts/infer.py --image ../04_data/sample_inputs/road2.png --no-tts
```

**Output:**

- `05_results/figures/outputs/road_HHMMSS.jpg` (annotated image with detections & guidance)
- Terminal: Navigation instruction + TTS audio

#### Video Mode

Process video files with adaptive frame sampling and temporal reasoning:

```bash
# Standard video processing (outputs to 05_results/figures/outputs/)
python scripts/infer.py --source ../04_data/sample_inputs/sample1.mp4

# Without audio
python scripts/infer.py --source ../04_data/sample_inputs/sample1.mp4 --no-tts

# Custom frame sampling interval (default 300ms)
python scripts/infer.py --source ../04_data/sample_inputs/sample1.mp4 --interval 200

# Save individual frames (every 5th sampled frame)
python scripts/infer.py --source ../04_data/sample_inputs/sample1.mp4 --save-frames
```

**Features:**

- Adaptive frame sampling (faster on hazards, slower on safe scenes)
- Asynchronous depth & detection (parallel processing)
- Multi-line caption wrapping
- FPS & latency metrics overlay
- Temporal smoothing (reduces flicker)

**Output:**

- Video: `05_results/figures/outputs/output.mp4` or `.avi`
- Frames: `05_results/figures/outputs/frame_NNNN.jpg` (if `--save-frames`)

#### Webcam (Real-Time)

Launch live demo with real-time audio guidance:

```bash
python scripts/demo.py

# Without audio
python scripts/demo.py --no-tts
```

Press `q` to quit.

---

## Training Commands

### YOLO Fine-Tuning on Indian Driving Dataset (IDD)

```bash
# Fine-tune YOLOv8n (nano) on IDD
python scripts/train_yolo.py \
    --data ../04_data/idd_data.yaml \
    --epochs 50 \
    --batch-size 16 \
    --img-size 640

# Options:
#   --model: base model (default: yolov8n.pt)
#           choices: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
#   --epochs: training epochs (default: 50)
#   --batch-size: batch size (default: 16)
#   --device: GPU ID (default: 0) or -1 for CPU
```

**Requirements:**

- IDD dataset in YAML format at `04_data/idd_data.yaml`
- Training outputs saved to `05_results/training/`

**Output:**

- Best weights: `05_results/training/yolo_idd_training/weights/best.pt`
- Training logs: `05_results/training/yolo_idd_training/`

---

## Evaluation Commands

### Latency & Flicker Rate Benchmark

```bash
python scripts/eval.py \
    --source ../04_data/sample_inputs/sample1.mp4 \
    --type latency \
    --max-frames 60
```

Measures:

- **Latency (ms):** Average inference time per frame
- **Flicker Rate:** Fraction of frames with instruction changes
- **FPS:** Frames processed per second

### Accuracy Ablation Study

```bash
python scripts/eval.py \
    --source ../04_data/sample_inputs/sample1.mp4 \
    --type accuracy \
    --max-frames 60
```

Measures:

- **Hazard Detection Rate:** Recall of true close-range hazards
- **False Alarm Rate:** % of safe frames with false emergency commands
- **Distance Accuracy:** Correct classification into distance bins
- **Walkable IoU:** Corridor detection accuracy vs. baseline

---

## Project Structure

```
03_code/
├── README.md                           # This file
├── requirements.txt                    # Pinned dependencies
│
├── src/                                # Core source code
│   ├── __init__.py
│   ├── main.py                         # Main entry point
│   ├── models/                         # ML modules
│   │   ├── detector.py                 # YOLOv8 object detection
│   │   ├── depth_estimator.py          # MiDaS monocular depth
│   │   ├── spatial_reasoning.py        # FOV-aware spatial reasoning
│   │   ├── tracker.py                  # DeepSORT tracking
│   │   ├── temporal_reasoner.py        # Motion & time-to-collision
│   │   ├── scene_memory.py             # Cost map & occupancy grid
│   │   ├── navigation_planner.py       # 8-layer rule-based planner
│   │   ├── frame_sampler.py            # Adaptive frame sampling
│   │   ├── metrics.py                  # Runtime statistics
│   │   ├── vlm_reasoner.py             # BLIP VQA validation (image mode)
│   │   ├── road_detector.py            # Walkable corridor detection
│   │   ├── corridor_estimator.py       # Safe path geometry
│   │   ├── occupancy_grid.py           # Spatial probability
│   │   └── weights/
│   │       └── idd_best.pt             # YOLOv8-IDD model weights
│   ├── caption/                        # Text generation
│   │   └── temporal_caption.py         # Hazard grouping & smoothing
│   ├── tts/                            # Text-to-speech
│   │   └── event_speaker.py            # Non-blocking audio output
│   └── utils/
│       └── visualize.py                # Bounding box & overlay rendering
│
├── scripts/                            # Runnable entry points
│   ├── infer.py                        # Inference (image/video/webcam)
│   ├── demo.py                         # Real-time webcam demo
│   ├── eval.py                         # Latency/accuracy benchmarks
│   └── train_yolo.py                   # YOLO fine-tuning
│
├── configs/                            # Configuration files
│   ├── default.yaml                    # Default settings
│   ├── infer.yaml                      # Inference settings
│   ├── train.yaml                      # Training hyperparameters
│   └── eval.yaml                       # Evaluation settings
│
├── MiDAS/                              # Depth estimation (git submodule)
│   ├── midas/
│   │   ├── base_model.py
│   │   ├── dpt_depth.py
│   │   ├── model_loader.py
│   │   └── backbones/
│   └── weights/
│       └── dpt_levit_224.pt
│
└── outputs/                            # Generated results (created at runtime)
    ├── output.mp4 / output.avi         # Processed video
    ├── frame_NNNN.jpg                  # Sampled frames (if --save-frames)
    └── image_HHMMSS.jpg                # Annotated images (image mode)
```

---

## Configuration

All configuration files are in `configs/` and use YAML format.

### Key Configurable Parameters

**Inference (`configs/infer.yaml`):**

```yaml
FRAME_SAMPLE_INTERVAL_MS: 300 # Frame sampling rate (ms)
RESIZE_SCALE: 0.6 # Processing resolution scale
USE_TTS: true # Enable audio output
DISPLAY_METRICS: true # Show latency/FPS overlay
```

**Detection (`configs/default.yaml`):**

```yaml
DETECTION_CONF_THRESH: 0.2 # Confidence threshold
FOV_AWARE_THRESHOLDS: true # Use FOV-aware distance classification
VERY_CLOSE_THRESHOLD: 0.60 # Depth threshold for "very close"
```

**Temporal (`configs/default.yaml`):**

```yaml
TEMPORAL_SMOOTHING_WINDOW: 3 # 3-frame smoothing
TEMPORAL_DECAY_ALPHA: 0.95 # Stale hazard decay rate
```

For complete configuration options, see the config files.

---

## Pipeline Architecture

```
INPUT: Video Frame / Image (640×384)
    │
    ├─→ [1] Object Detection (YOLOv8-IDD)
    │       └─→ Detects: vehicles, pedestrians, animals, autorickshaws
    │
    ├─→ [2] Depth Estimation (MiDaS DPT-LeViT)
    │       └─→ Normalized depth [0, 1]
    │
    ├─→ [3] Spatial Reasoning
    │       ├─→ FOV-aware distance classification
    │       ├─→ Direction (left/center/right)
    │       └─→ Risk scoring
    │
    ├─→ [4] Object Tracking (DeepSORT)
    │       └─→ Persistent track IDs + motion vectors
    │
    ├─→ [5] Temporal Reasoning
    │       ├─→ Time-to-collision estimation
    │       └─→ Stale hazard decay
    │
    ├─→ [6] Scene Memory & Navigation Planning
    │       ├─→ Cost map computation
    │       ├─→ Corridor detection
    │       └─→ 8-layer rule-based planner
    │
    └─→ [7] Caption Generation, VLM Refinement & TTS
            ├─→ Temporal caption (grouping + smoothing)
            ├─→ VLM visual grounding (image mode only)
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
Rule 6: Road penalty (center > 1.2)  → EDGE LEFT/RIGHT (info)
Rule 7: Distant hazards only         → CONTINUE FORWARD (info)
Rule 8: Corridor-based routing       → FOLLOW CORRIDOR (info)
```

---

## Troubleshooting

### Common Issues

**Q: Model weights not found**

```
FileNotFoundError: Required model weight not found: models/weights/idd_best.pt
```

**A:** Download model weights using the Quick Start instructions above.

**Q: MiDAS import error**

```
ModuleNotFoundError: No module named 'MiDAS'
```

**A:** Clone the MiDAS repository:

```bash
git clone https://github.com/isl-org/MiDAS.git
```

**Q: Low FPS / slow inference**

- Use GPU: ensure `torch.cuda.is_available()` returns `True`
- Reduce `RESIZE_SCALE` in `configs/default.yaml` (default 0.6)
- Increase `FRAME_SAMPLE_INTERVAL_MS` (larger = fewer frames processed)
- Disable TTS: use `--no-tts` flag

**Q: No audio output**

- Check TTS installation: `pip install pyttsx3`
- Verify system audio is not muted
- Use `--no-tts` to disable and confirm video output works

**Q: Video codec issues**

- macOS prefers `avc1` (MP4); falls back to XVID (AVI) if unavailable
- Windows/Linux may require codec installation

---

## Ablation Study Results

See `05_results/ablations.csv` for detailed component impact analysis:

| Component             | Hazard Recall | Distance Accuracy | Latency (ms) | Flicker  |
| --------------------- | ------------- | ----------------- | ------------ | -------- |
| Full Pipeline         | 97.9%         | 92.8%             | 84.6         | 0.39     |
| No Temporal Smoothing | 98.9%         | 88.2%             | 79.3         | 0.29     |
| No FOV-Aware Depth    | 96.7%         | **0.0%**          | 84.3         | 0.17     |
| No Road Detection     | 100.0%        | 86.9%             | 81.4         | 0.23     |
| No DeepSORT Tracking  | 100.0%        | 88.7%             | 80.7         | **0.48** |

**Key Finding:** FOV-aware spatial reasoning is critical (distance accuracy drops to 0% without it).

---

## References

Key research papers that influenced this project:

1. **Towards Blind and Low-Vision Accessibility of Lightweight VLMs**
    - Lightweight VLMs for accessibility
    - Prompt design for navigation

2. **End-to-End Navigation with Vision-Language Models (VLMnav)**
    - Depth-informed navigation strategies
    - Discrete actionable guidance

3. **VLM-Grounder: A VLM Agent for Zero-Shot 3D Visual Grounding**
    - Multi-frame aggregation for improved robustness
    - Grounding without 3D reconstruction

4. **VizWiz: Nearly Real-Time Answers to Visual Questions**
    - Real-world assistive vision system challenges
    - User-centric evaluation metrics

---

## License

This project is part of CS F425 (Deep Learning) at BITS Pilani.

## Authors

- **Ishika Goyal** (2022B3A70549P) - Research, experimentation, ablation studies
- **Udit Agarwal** (2023A8PS0708P) - Core pipeline, FOV-aware spatial reasoning, YOLO fine-tuning
- **Mohul Batra** (2023A1PS0164P) - Tracking, temporal reasoning, demo preparation

**Mentor:** Prashant Trivedi  
**Faculty:** Pratik Narang
