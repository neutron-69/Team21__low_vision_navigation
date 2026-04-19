#!/usr/bin/env python3
"""
Training Script
Handles YOLO fine-tuning on Indian Driving Dataset (IDD).

Usage:
    python train_yolo.py --data ../../04_data/idd_data.yaml --epochs 50
"""

import argparse
from pathlib import Path
from importlib import import_module

try:
    YOLO = import_module("ultralytics").YOLO
except Exception:
    print("ERROR: ultralytics not installed. Run: pip install -r ../requirements.txt")
    raise


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on IDD dataset")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device (0=GPU, -1=CPU)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", 
                        help="Base model (default: nano)")
    
    args = parser.parse_args()
    
    print("=" * 65)
    print("  YOLO FINE-TUNING")
    print("=" * 65)
    print(f"  Dataset: {args.data}")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print()
    
    model = YOLO(args.model)
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        save=True,
        project="../05_results/training",
        name="yolo_idd_training",
        patience=10,
    )
    
    print("\n[DONE] Training complete.")
    print(f"[SAVED] Results → ../05_results/training/")


if __name__ == "__main__":
    main()
