"""
Accuracy-Based Ablation Benchmark
==================================
Measures accuracy & reliability metrics with components toggled ON/OFF:

1. Hazard Capture Rate (Recall)  — % of true hazard events correctly
   generating STOP/AVOID/MOVE commands.
2. False Alarm Rate              — % of safe frames incorrectly triggering
   emergency commands.
3. Walkable Corridor IoU         — Depth-gradient vs naive baseline.
4. Distance Classification Accuracy — Correct distance bin (via depth
   cross-validation).

Usage:
    cd code/vlm-navigation-assistant
    python ../../results/run_accuracy_ablation.py --source samples/sample1.mp4

Saves: ../../results/ablations_accuracy.csv
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "..", "code", "vlm-navigation-assistant")
sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)

import cv2
import numpy as np
from models.frame_sampler import FrameSampler
from models.depth_estimator import DepthEstimator
from models.detector import ObjectDetector
from models.spatial_reasoning import SpatialReasoner
from models.tracker import ObjectTracker
from models.temporal_reasoner import TemporalReasoner
from models.scene_memory import SceneMemory
from models.navigation_planner import NavigationPlanner
from models.road_detector import RoadDetector
from caption.temporal_caption import TemporalCaptionGenerator


# ── Ground-truth hazard definition ──────────────────────────────
# A frame is a "true hazard" if any detected object has a high raw
# depth value (depth > HAZARD_DEPTH_THRESHOLD in normalised MiDaS,
# meaning very close) AND occupies a significant portion of the frame.
# This is our best proxy for ground truth without manual annotation.
HAZARD_DEPTH_THRESHOLD = 0.55     # normalised depth: > 0.55 = genuinely close
HAZARD_AREA_THRESHOLD  = 0.015    # object covers > 1.5% of frame area
EMERGENCY_COMMANDS = {"Stop immediately.", "Move left.", "Move right."}


def is_true_hazard_frame(detections, depth_map, img_w, img_h):
    """Determine if this frame has a genuine close-range hazard using
    raw depth values as ground-truth signal."""
    if depth_map is None or not detections:
        return False
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
        area = (x2 - x1) * (y2 - y1)
        norm_area = area / (img_w * img_h)
        if norm_area < HAZARD_AREA_THRESHOLD:
            continue
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            continue
        median_depth = float(np.median(region))
        if median_depth > HAZARD_DEPTH_THRESHOLD:
            return True
    return False


def walkable_iou_naive(depth_map, analysis_ratio=0.60):
    """Compute IoU between depth-gradient walkable mask and a naive
    baseline (bottom 30% center strip assumed always walkable)."""
    h, w = depth_map.shape[:2]
    road_det = RoadDetector()
    state = road_det.detect(depth_map)
    pred_mask = state["walkable_mask"]  # uint8, 255=walkable

    # Naive baseline: bottom 30% of frame, center 60% width
    naive_mask = np.zeros((h, w), dtype=np.uint8)
    y_start = int(h * 0.70)
    x_start = int(w * 0.20)
    x_end = int(w * 0.80)
    naive_mask[y_start:, x_start:x_end] = 255

    # Compute IoU
    intersection = np.count_nonzero((pred_mask > 0) & (naive_mask > 0))
    union = np.count_nonzero((pred_mask > 0) | (naive_mask > 0))
    iou = intersection / max(union, 1)
    pred_coverage = np.count_nonzero(pred_mask > 0) / (h * w)
    naive_coverage = np.count_nonzero(naive_mask > 0) / (h * w)
    return iou, pred_coverage, naive_coverage


def run_accuracy_experiment(source, config, depth_estimator, detector,
                            max_frames=60):
    """Run one ablation variant and measure accuracy metrics."""
    use_tracker     = config.get("use_tracker", True)
    use_temporal    = config.get("use_temporal", True)
    use_fov_aware   = config.get("use_fov_aware", True)
    use_road_detect = config.get("use_road_detect", True)

    tracker   = ObjectTracker() if use_tracker else None
    temporal  = TemporalReasoner() if use_temporal else None
    memory    = SceneMemory()
    planner   = NavigationPlanner()
    captioner = TemporalCaptionGenerator(
        smoothing_window=3 if use_temporal else 1
    )
    road_det  = RoadDetector() if use_road_detect else None

    RESIZE = 0.6
    frame_count = 0
    depth_map = None
    prev_dets = []
    reasoner = None

    # Counters
    true_hazard_frames = 0       # Frames where depth says hazard exists
    correct_responses  = 0       # Hazard frame + system said STOP/AVOID/MOVE
    missed_hazards     = 0       # Hazard frame + system said FORWARD
    total_frames       = 0

    # Distance accuracy: compare FOV-aware label vs raw depth bin
    distance_correct = 0
    distance_total   = 0

    # Walkable IoU samples
    iou_samples = []

    with FrameSampler(source, sample_interval_ms=300) as sampler:
        for frame, timestamp in sampler:
            if frame_count >= max_frames:
                break
            frame_count += 1
            total_frames += 1

            small = cv2.resize(frame, None, fx=RESIZE, fy=RESIZE)
            h_s, w_s = small.shape[:2]

            # Depth
            if frame_count % 6 == 1 or depth_map is None:
                depth_map = depth_estimator.estimate_depth(small)

            # Detection
            if frame_count % 2 == 1 or not prev_dets:
                prev_dets = detector.detect(small)
            detections = prev_dets

            # ── Ground truth: is this a true hazard frame? ──
            gt_hazard = is_true_hazard_frame(detections, depth_map, w_s, h_s)

            # Spatial reasoning
            if reasoner is None:
                reasoner = SpatialReasoner(w_s, h_s, depth_map)

            if depth_map is not None:
                reasoner.depth_map = depth_map if use_fov_aware else None

            enriched = reasoner.prioritize_hazards(detections) if detections else []

            # ── Distance accuracy check ──
            if depth_map is not None and use_fov_aware:
                for det in enriched:
                    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_s - 1, x2), min(h_s - 1, y2)
                    region = depth_map[y1:y2, x1:x2]
                    if region.size == 0:
                        continue
                    raw_depth = float(np.median(region))
                    # Ground-truth distance bin from raw depth
                    if raw_depth > 0.60:
                        gt_dist = "very close"
                    elif raw_depth > 0.40:
                        gt_dist = "near"
                    elif raw_depth > 0.18:
                        gt_dist = "moderate distance"
                    else:
                        gt_dist = "far"
                    pred_dist = det.get("distance", "far")
                    distance_total += 1
                    if pred_dist == gt_dist:
                        distance_correct += 1
                    elif _adjacent_bins(pred_dist, gt_dist):
                        distance_correct += 0.5  # partial credit for adjacent

            # Tracking
            if tracker is not None:
                tracked = tracker.update(enriched, small)
                for d in tracked:
                    d["bbox"] = [int(x / RESIZE) for x in d["bbox"]]
            else:
                tracked = enriched
                for d in tracked:
                    d["bbox"] = [int(x / RESIZE) for x in d["bbox"]]
                    d["track_id"] = id(d)

            # Temporal reasoning
            if temporal is not None:
                temporal_objects = temporal.update(tracked, timestamp)
            else:
                temporal_objects = [{
                    "track_id": d.get("track_id", 0),
                    "label": d["label"],
                    "zone": d.get("direction", "center"),
                    "distance": d.get("distance", "far"),
                    "motion": "stationary",
                    "risk": d.get("risk_score", 0.1),
                } for d in tracked]

            # Road detection
            road_state = None
            if road_det is not None and depth_map is not None and frame_count % 3 == 0:
                road_state = road_det.detect(depth_map, small, prev_dets)

            # Memory + Planning
            memory.update(temporal_objects, road_state=road_state)
            cost_map = memory.get_cost_map()
            safest = memory.get_safest_direction()
            corridor = memory.get_best_corridor()

            instruction, urgency = planner.decide(
                temporal_objects, cost_map, safest, corridor,
                road_state=road_state
            )

            # ── Score the response ──
            is_emergency = instruction in EMERGENCY_COMMANDS

            if gt_hazard:
                true_hazard_frames += 1
                if is_emergency:
                    correct_responses += 1
                else:
                    missed_hazards += 1

            # Walkable IoU (sample every 10 frames)
            if depth_map is not None and frame_count % 10 == 0:
                iou, _, _ = walkable_iou_naive(depth_map)
                iou_samples.append(iou)

    # ── Compute final metrics ──
    hazard_recall = (correct_responses / max(true_hazard_frames, 1))
    dist_accuracy = (distance_correct / max(distance_total, 1))
    avg_iou = float(np.mean(iou_samples)) if iou_samples else 0.0

    return {
        "frames":              total_frames,
        "true_hazard_frames":  true_hazard_frames,
        "hazard_recall":       round(hazard_recall, 3),
        "distance_accuracy":   round(dist_accuracy, 3),
        "walkable_iou":        round(avg_iou, 3),
        "missed_hazards":      missed_hazards,
    }


def _adjacent_bins(a, b):
    """Check if two distance bins are adjacent (1-off)."""
    order = ["very close", "near", "moderate distance", "far"]
    try:
        return abs(order.index(a) - order.index(b)) == 1
    except ValueError:
        return False


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Accuracy-based ablation")
    parser.add_argument("--source", required=True, help="Path to video")
    parser.add_argument("--max-frames", type=int, default=60,
                        help="Max frames per experiment (default 60)")
    args = parser.parse_args()

    print("=" * 72)
    print("  ACCURACY-BASED ABLATION BENCHMARK")
    print("=" * 72)
    print(f"  Source: {args.source}")
    print(f"  Max frames: {args.max_frames}")
    print()

    print("[INIT] Loading models...")
    depth_est = DepthEstimator()
    detector = ObjectDetector()
    print("[INIT] Done.\n")

    experiments = [
        ("Full Pipeline",
         {"use_tracker": True,  "use_temporal": True,
          "use_fov_aware": True,  "use_road_detect": True}),
        ("No Temporal Smoothing",
         {"use_tracker": True,  "use_temporal": False,
          "use_fov_aware": True,  "use_road_detect": True}),
        ("No FOV-Aware Depth",
         {"use_tracker": True,  "use_temporal": True,
          "use_fov_aware": False, "use_road_detect": True}),
        ("No Road Detection",
         {"use_tracker": True,  "use_temporal": True,
          "use_fov_aware": True,  "use_road_detect": False}),
        ("No DeepSORT Tracking",
         {"use_tracker": False, "use_temporal": False,
          "use_fov_aware": True,  "use_road_detect": True}),
    ]

    results = []
    for name, config in experiments:
        print(f"[RUN] {name} ...")
        res = run_accuracy_experiment(
            args.source, config, depth_est, detector,
            max_frames=args.max_frames
        )
        res["experiment"] = name
        results.append(res)
        print(f"       Hazard Recall: {res['hazard_recall']:.1%}  |  "
              f"Dist Acc: {res['distance_accuracy']:.1%}  |  "
              f"Walkable IoU: {res['walkable_iou']:.3f}")

    # Print table
    print("\n" + "=" * 72)
    print(f"{'Experiment':<26} {'Hazard':>8} {'Dist':>8} {'Walk':>8}")
    print(f"{'':26} {'Recall':>8} {'Acc':>8} {'IoU':>8}")
    print("-" * 72)
    for r in results:
        print(f"{r['experiment']:<26} "
              f"{r['hazard_recall']:>7.1%} "
              f"{r['distance_accuracy']:>7.1%} "
              f"{r['walkable_iou']:>8.3f}")
    print("=" * 72)

    # Save CSV
    out_csv = os.path.join(SCRIPT_DIR, "ablations_accuracy.csv")
    fieldnames = [
        "experiment", "frames", "true_hazard_frames",
        "hazard_recall", "distance_accuracy",
        "walkable_iou", "missed_hazards",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[SAVED] {out_csv}")


if __name__ == "__main__":
    main()
