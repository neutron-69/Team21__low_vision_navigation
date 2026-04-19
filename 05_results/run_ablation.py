"""
Ablation Benchmark Script
=========================
Runs your pipeline on a sample video with individual components toggled OFF
to measure their impact on latency and instruction stability (flicker rate).

Usage:
    cd code/vlm-navigation-assistant
    python ../../results/run_ablation.py --source samples/sample1.mp4

Outputs: prints a results table and saves to ../../results/ablations_measured.csv
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

# Add the project root to path so imports work
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


# ─────────────────────────────────────────────────
# Flicker rate: fraction of frames where the
# navigation instruction changed unnecessarily.
# ─────────────────────────────────────────────────
def compute_flicker_rate(instructions):
    if len(instructions) < 2:
        return 0.0
    changes = sum(1 for i in range(1, len(instructions))
                  if instructions[i] != instructions[i - 1])
    return changes / (len(instructions) - 1)


# ─────────────────────────────────────────────────
# Single ablation experiment run
# ─────────────────────────────────────────────────
def run_experiment(source, config, depth_estimator, detector, max_frames=60):
    """
    config dict keys (all default True):
        use_tracker       – enable DeepSORT tracking
        use_temporal      – enable temporal reasoning
        use_fov_aware     – enable FOV-aware depth thresholds
        use_road_detect   – enable RoadDetector
    """
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
    instructions = []
    latencies = []

    with FrameSampler(source, sample_interval_ms=300) as sampler:
        for frame, timestamp in sampler:
            if frame_count >= max_frames:
                break

            t0 = time.monotonic()
            frame_count += 1

            small = cv2.resize(frame, None, fx=RESIZE, fy=RESIZE)
            h_s, w_s = small.shape[:2]

            # Depth (every 6th frame or first)
            if frame_count % 6 == 1 or depth_map is None:
                depth_map = depth_estimator.estimate_depth(small)

            # Detection (every 2nd frame or first)
            if frame_count % 2 == 1 or not prev_dets:
                prev_dets = detector.detect(small)

            detections = prev_dets

            # Spatial reasoning
            if reasoner is None:
                if depth_map is None:
                    depth_map = depth_estimator.estimate_depth(small)
                reasoner = SpatialReasoner(w_s, h_s, depth_map)

            if depth_map is not None:
                if use_fov_aware:
                    reasoner.depth_map = depth_map
                else:
                    # Disable FOV-aware: set depth_map to None so it
                    # falls back to simple perspective-based distance
                    reasoner.depth_map = None

            enriched = reasoner.prioritize_hazards(detections) if detections else []

            # Tracking
            if tracker is not None:
                tracked = tracker.update(enriched, small)
                for d in tracked:
                    d["bbox"] = [int(x / RESIZE) for x in d["bbox"]]
            else:
                tracked = enriched
                for d in tracked:
                    d["bbox"] = [int(x / RESIZE) for x in d["bbox"]]
                    d["track_id"] = id(d)  # fake unique ID

            # Temporal reasoning
            if temporal is not None:
                temporal_objects = temporal.update(tracked, timestamp)
            else:
                # No temporal: just map detections directly
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

            _, full_caption = captioner.generate(
                temporal_objects, instruction, urgency,
                road_state=road_state
            )

            instructions.append(instruction)
            elapsed = time.monotonic() - t0
            latencies.append(elapsed)

    avg_latency_ms = (np.mean(latencies) * 1000) if latencies else 0
    flicker = compute_flicker_rate(instructions)

    return {
        "frames": frame_count,
        "avg_latency_ms": round(avg_latency_ms, 1),
        "flicker_rate": round(flicker, 3),
    }


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ablation benchmark")
    parser.add_argument("--source", required=True, help="Path to video file")
    parser.add_argument("--max-frames", type=int, default=40,
                        help="Max frames per experiment (default 40)")
    args = parser.parse_args()

    print("=" * 65)
    print("  ABLATION BENCHMARK")
    print("=" * 65)
    print(f"  Source: {args.source}")
    print(f"  Max frames per experiment: {args.max_frames}")
    print()

    # Load models once (shared across experiments)
    print("[INIT] Loading shared models (detector + depth)...")
    depth_est = DepthEstimator()
    detector = ObjectDetector()
    print("[INIT] Done.\n")

    experiments = [
        ("Full Pipeline",          {"use_tracker": True,  "use_temporal": True,  "use_fov_aware": True,  "use_road_detect": True}),
        ("No Temporal Smoothing",  {"use_tracker": True,  "use_temporal": False, "use_fov_aware": True,  "use_road_detect": True}),
        ("No FOV-Aware Depth",     {"use_tracker": True,  "use_temporal": True,  "use_fov_aware": False, "use_road_detect": True}),
        ("No Road Detection",      {"use_tracker": True,  "use_temporal": True,  "use_fov_aware": True,  "use_road_detect": False}),
        ("No DeepSORT Tracking",   {"use_tracker": False, "use_temporal": False, "use_fov_aware": True,  "use_road_detect": True}),
    ]

    results = []
    for name, config in experiments:
        print(f"[RUN] {name} ...")
        res = run_experiment(args.source, config, depth_est, detector,
                             max_frames=args.max_frames)
        res["experiment"] = name
        results.append(res)
        print(f"       → {res['frames']} frames | "
              f"Latency: {res['avg_latency_ms']} ms | "
              f"Flicker: {res['flicker_rate']:.2f}")

    # Print table
    print("\n" + "=" * 65)
    print(f"{'Experiment':<28} {'Latency (ms)':>13} {'Flicker Rate':>13}")
    print("-" * 65)
    for r in results:
        print(f"{r['experiment']:<28} {r['avg_latency_ms']:>13.1f} {r['flicker_rate']:>13.3f}")
    print("=" * 65)

    # Save CSV
    out_csv = os.path.join(SCRIPT_DIR, "ablations_measured.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment", "frames", "avg_latency_ms", "flicker_rate"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[SAVED] {out_csv}")


if __name__ == "__main__":
    main()
