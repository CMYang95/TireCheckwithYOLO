import argparse
from pathlib import Path


def _import_yolo():
    """
    優先使用 pip 安裝的 ultralytics，避免被本專案的 `ultralytics/` 資料夾遮蔽。
    """
    import sys

    repo_root = Path(__file__).resolve().parent
    for p in (str(repo_root), str(repo_root / "ultralytics")):
        while p in sys.path:
            sys.path.remove(p)

    from ultralytics import YOLO  # noqa: E402

    return YOLO


def _parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 (Ultralytics) on Tire_dataset.")
    p.add_argument(
        "--data",
        type=str,
        default=str(Path("Tire_dataset") / "tiredata.yaml"),
        help="Dataset YAML path (relative or absolute).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=str(Path("yolov8n.pt")),
        help="Model to start from (e.g. yolov8n.pt or a .yaml).",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=-1, help="Batch size (-1 = auto).")
    p.add_argument("--device", type=str, default=None, help="e.g. 0, 0,1, cpu")
    p.add_argument("--project", type=str, default="runs")
    p.add_argument("--name", type=str, default="detect/train_tire")
    return p.parse_args()


def main():
    args = _parse_args()

    YOLO = _import_yolo()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()

