"""
Legacy demo script.

建議改用專案根目錄的 predict.py：
  python predict.py --weights runs/detect/train3/weights/best.pt --source Tire_dataset/images/tire22.jpg
"""

from pathlib import Path

import sys

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "ultralytics"))

from ultralytics import YOLO


def main():
    weights = repo_root / "runs" / "detect" / "train3" / "weights" / "best.pt"
    source = repo_root / "Tire_dataset" / "images" / "tire22.jpg"
    model = YOLO(str(weights))
    model.predict(source=str(source), save=True, conf=0.2)


if __name__ == "__main__":
    main()

# from ultralytics import YOLO

# # Load a model
# model = YOLO(r"D:\TireCheck\runs\detect\train\weights\best.pt")  # load a custom model
# print(model)