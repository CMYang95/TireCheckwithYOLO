import argparse
from pathlib import Path


def _import_yolo():
    """
    優先使用 pip 安裝的 ultralytics。
    由於本專案根目錄有一個 `ultralytics/` 資料夾（原始碼），會遮蔽 site-packages 的套件。
    """
    import sys

    repo_root = Path(__file__).resolve().parent
    # 移除會造成遮蔽的路徑
    for p in (str(repo_root), str(repo_root / "ultralytics")):
        while p in sys.path:
            sys.path.remove(p)

    from ultralytics import YOLO  # noqa: E402

    return YOLO


def _parse_args():
    p = argparse.ArgumentParser(description="Run YOLOv8 inference and save results.")
    p.add_argument(
        "--weights",
        type=str,
        default=str(Path("runs") / "detect" / "train3" / "weights" / "best.pt"),
        help="Path to model weights (.pt).",
    )
    p.add_argument(
        "--source",
        type=str,
        default=str(Path("Tire_dataset") / "images"),
        help="Image/video/folder path, glob, webcam id, or URL.",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--project", type=str, default=str(Path("runs") / "detect"))
    p.add_argument("--name", type=str, default="predict_tire")
    p.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-txt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-conf", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main():
    args = _parse_args()

    YOLO = _import_yolo()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )


if __name__ == "__main__":
    main()

