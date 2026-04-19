#!/usr/bin/env python3
"""
Inference Script
Runs the navigation pipeline on an image or video file.

Usage:
    python infer.py --image ../../04_data/sample_inputs/road.jpg
    python infer.py --source ../../04_data/sample_inputs/video.mp4
    python infer.py --source 0  # webcam
"""

import sys
import os
import argparse
from pathlib import Path

# Add 03_code root to path for src package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import run, run_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run navigation inference")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--source", type=str, help="Path to video or camera ID (0 for webcam)")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech")
    parser.add_argument("--interval", type=int, default=300, help="Frame sampling interval (ms)")
    parser.add_argument("--save-frames", action="store_true", help="Save individual frames")
    
    args = parser.parse_args()
    
    if args.image:
        run_image(args.image, use_tts=not args.no_tts)
    elif args.source:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
        run(source, use_tts=not args.no_tts, sample_interval_ms=args.interval, 
            save_frames=args.save_frames)
    else:
        parser.print_help()
