"""
Legacy YOLOv5 torch.hub demo.

這份檔案容易造成依賴混亂（YOLOv5 hub vs YOLOv8 Ultralytics）。
建議統一用 predict.py（YOLOv8）。
"""

from pathlib import Path

import sys

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "ultralytics"))

from ultralytics import YOLO


def main():
    weights = repo_root / "runs" / "detect" / "train6" / "weights" / "tiredata.pt"
    source = repo_root / "Tire_dataset" / "images" / "tire4.jpg"
    model = YOLO(str(weights))
    model.predict(source=str(source), save=True, conf=0.5)


if __name__ == "__main__":
    main()
