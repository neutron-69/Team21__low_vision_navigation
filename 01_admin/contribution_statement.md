# Contribution Statement

## Team 21: Vision-Language Models for Low-Vision Navigation Assistance

### Team Members:

- **Ishika Goyal (2022B3A70549P)**
- **Udit Agarwal (2023A8PS0708P)**
- **Mohul Batra (2023A1PS0164P)**

---

## Individual Contributions

### Ishika Goyal - Reading, Experimentation, Report Writing

Ishika led the research and evaluation phase. She conducted extensive literature review of relevant papers (VLM-Grounder, VizWiz, VLMnav) and prior work in assistive vision systems. During the experimentation phase, she along with mohul designed and executed all ablation studies (6 experiment configurations in `results/run_ablation.py`), analyzing the impact of each component on latency, flicker rate, and accuracy metrics. She identified critical findings such as the FOV-aware depth module's importance (distance accuracy drops to 0% without it) and quantified the trade-offs between temporal smoothing and instruction stability. Ishika also documented comprehensive prior work analysis (`claims/prior_work_basis.md`) and synthesized project findings into clear technical narratives for the final report.

### Udit Agarwal - Coding & System Integration

Udit architected and implemented the core pipeline system. He built the main entry point (`main.py`) integrating six major modules: object detection (YOLOv8 fine-tuning), depth estimation (MiDaS), spatial reasoning, object tracking (DeepSORT), temporal reasoning, and navigation planning. Udit implemented the **FOV-aware spatial reasoning module** (`models/spatial_reasoning.py`), which combines depth maps with vertical position and bounding box size to classify objects into distance bands—this is the critical module identified in ablations. He also engineered the **IDD dataset pipeline** (`data/data_description.md`), converting XML annotations to YOLO format and managing train/val/test splits across 46,588 images. Additional implementations include the depth estimator, navigation planner (8-layer rule system), and TTS gating logic for reducing audio cognitive overload.

### Mohul Batra - Coding, Demo Preparation, and Presentation

Mohul focused on making the system accessible and demonstrable. He implemented support modules including the **object tracker** (DeepSORT integration), **temporal reasoning** (motion vector computation and time-to-collision estimation), **scene memory** (occupancy grid and corridor detection), and the **TTS speaker** system. Mohul prepared comprehensive demo instructions (`demo/demo_instructions.md`) with step-by-step setup procedures and example runs. He created the user-facing video visualization pipeline with text wrapping, overlay annotations, and hazard-priority highlighting. Mohul also managed project documentation, including README files explaining the complete pipeline architecture, and prepared the system for live demonstration to blind/low-vision user studies.

---

All members contributed equally to problem definition, design decisions, and integration testing. Division of effort reflects individual expertise areas and project timeline constraints.
