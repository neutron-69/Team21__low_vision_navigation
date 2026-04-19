#!/usr/bin/env python3
"""
Demo Script
Launches real-time webcam-based navigation assistant with audio guidance.

Usage:
    python demo.py
    python demo.py --no-tts
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch live navigation demo")
    parser.add_argument("--no-tts", action="store_true", help="Disable audio output")
    
    args = parser.parse_args()
    
    print("🎥 Launching live webcam navigation assistant...")
    print("   Press 'q' to quit")
    
    run(source=0, use_tts=not args.no_tts, sample_interval_ms=300, save_frames=False)
