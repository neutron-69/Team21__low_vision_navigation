#!/usr/bin/env python3
"""
Evaluation Script
Runs ablation studies and accuracy metrics on test videos.

Usage:
    python eval.py --source ../../04_data/sample_inputs/video.mp4 --type latency
    python eval.py --source ../../04_data/sample_inputs/video.mp4 --type accuracy
"""

import sys
import os
import argparse
import csv
import time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.models.frame_sampler import FrameSampler
from src.models.depth_estimator import DepthEstimator
from src.models.detector import ObjectDetector
from src.models.spatial_reasoning import SpatialReasoner
from src.models.tracker import ObjectTracker
from src.models.temporal_reasoner import TemporalReasoner
from src.models.scene_memory import SceneMemory
from src.models.navigation_planner import NavigationPlanner
from src.models.road_detector import RoadDetector
from src.caption.temporal_caption import TemporalCaptionGenerator


def compute_flicker_rate(instructions):
    """Compute flicker: % of frames where instruction changes unnecessarily."""
    if len(instructions) < 2:
        return 0.0
    changes = sum(1 for i in range(1, len(instructions))
                  if instructions[i] != instructions[i - 1])
    return changes / (len(instructions) - 1)


def run_latency_benchmark(source, max_frames=60):
    """Benchmark latency and flicker rate with full pipeline."""
    depth_est = DepthEstimator()
    detector = ObjectDetector()
    tracker = ObjectTracker()
    temporal = TemporalReasoner()
    memory = SceneMemory()
    planner = NavigationPlanner()
    captioner = TemporalCaptionGenerator()
    road_det = RoadDetector()
    
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
            
            if frame_count % 6 == 1 or depth_map is None:
                depth_map = depth_est.estimate_depth(small)
            
            if frame_count % 2 == 1 or not prev_dets:
                prev_dets = detector.detect(small)
            
            detections = prev_dets
            
            if reasoner is None:
                if depth_map is None:
                    depth_map = depth_est.estimate_depth(small)
                reasoner = SpatialReasoner(w_s, h_s, depth_map)
            
            if depth_map is not None:
                reasoner.depth_map = depth_map
            
            enriched = reasoner.prioritize_hazards(detections) if detections else []
            tracked = tracker.update(enriched, small)
            for d in tracked:
                d["bbox"] = [int(x / RESIZE) for x in d["bbox"]]
            
            temporal_objects = temporal.update(tracked, timestamp)
            
            road_state = road_det.detect(depth_map, small, prev_dets) if frame_count % 3 == 0 else None
            
            memory.update(temporal_objects, road_state=road_state)
            cost_map = memory.get_cost_map()
            safest = memory.get_safest_direction()
            corridor = memory.get_best_corridor()
            
            instruction, urgency = planner.decide(
                temporal_objects, cost_map, safest, corridor,
                road_state=road_state
            )
            
            _, full_caption = captioner.generate(
                temporal_objects, instruction, urgency, road_state=road_state
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


def main():
    parser = argparse.ArgumentParser(description="Run evaluation benchmarks")
    parser.add_argument("--source", required=True, help="Path to video file")
    parser.add_argument("--type", default="latency", choices=["latency", "accuracy"],
                        help="Evaluation type")
    parser.add_argument("--max-frames", type=int, default=60, help="Max frames per test")
    args = parser.parse_args()
    
    print("=" * 65)
    print(f"  EVALUATION: {args.type.upper()}")
    print("=" * 65)
    print(f"  Source: {args.source}")
    print()
    
    if args.type == "latency":
        result = run_latency_benchmark(args.source, max_frames=args.max_frames)
        print(f"[RESULT] Frames: {result['frames']}")
        print(f"[RESULT] Avg Latency: {result['avg_latency_ms']:.1f} ms")
        print(f"[RESULT] Flicker Rate: {result['flicker_rate']:.3f}")
    
    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
