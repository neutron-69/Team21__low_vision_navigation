# Claimed Contributions

### What we reproduced
*   **Object Detection Baseline:** We successfully reproduced real-time 2D object detection using the base YOLOv8 architecture to extract bounding boxes and confidences.
*   **Monocular Depth Estimation:** We reproduced the zero-shot monocular depth estimation pipeline using the pre-trained MiDaS DPT-LeViT model to generate normalized disparity maps.
*   **Object Tracking:** We integrated and reproduced the core temporal tracking behaviors of the DeepSORT algorithm to maintain continuous IDs for obstacles across consecutive video frames.

### What we modified
*   **Domain-Specific Fine-Tuning:** We fine-tuned the YOLOv8 model on the Indian Driving Dataset (IDD) to shift the perception domain away from standard COCO classes toward unstructured traffic environments (detecting autorickshaws, unhelmeted riders, stray animals).
*   **FOV-Aware Spatial Fusion:** We built a custom Spatial Reasoner layer that calculates the median depth of an object by intersecting 2D bbox coordinates with the MiDaS depth map. We modified standard thresholding by implementing a Field-of-View (FOV)-aware scale that dynamically adjusts depth thresholds based on the object's vertical pixel location.
*   **Depth-Gradient Road Extraction:** Rather than using a standalone semantic segmentation network for road mapping, we designed a custom Walkable-Space Detector that extracts traversable corridor paths relying strictly on structural geometry (depth map gradients).
*   **Navigational Translation Layer:** We developed an 8-layer navigational rule planner (capable of mapping geometries to commands like STOP/AVOID) coupled with a smart Text-to-Speech (TTS) gating system that intelligently drops stale frames to preserve audio fluidity.

### What did not work
*   **Real-Time Sub-Frame VLM Analysis:** Our initial attempts to use a Vision-Language Model (BLIP) for frame-by-frame scene reasoning resulted in severe hardware bottlenecks, dropping throughput to <1 FPS. This forced us to decouple the VLM entirely from the live video loop and relegate it to an offline or static image-mode validation layer.
*   **Absolute Distance Calibration in Meters:** Because the monocular depth (MiDaS) model outputs normalized, scale-and-shift invariant representations, extracting accurate, absolute distance measurements (in meters) failed without an active stereo calibration matrix. Consequently, we abandoned absolute metrics in favor of logical relative proximity bands (Critical, Warning, Info) and motion-vector time-to-collision.

### What we believe is our contribution
*   **Integrated Cluttered-Environment Pipeline:** An end-to-end, asynchronous vision-to-audio pipeline specifically targeting the unique challenges of visually impaired navigation in chaotic, unstructured Indian street contexts. 
*   **Audio Cognitive-Load Reduction:** The synthesis of raw computer vision matrices into a smoothed, priority-gated navigational grammar. By decaying stale hazards and suppressing passive scene descriptions when urgent evasion commands are required, our TTS gating system actively reduces cognitive overwhelm for the end-user.
*   **Computational Efficiency in Pathing:** The development of a computationally lightweight logic layer that infers safe, walkable edge-corridors indirectly through relative depth map gradients, circumventing the need to run computationally expensive mesh or semantic classification networks in parallel alongside YOLO and MiDaS.
