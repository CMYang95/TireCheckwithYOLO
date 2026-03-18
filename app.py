from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image


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


@st.cache_resource(show_spinner=False)
def _load_model(weights_path: str):
    YOLO = _import_yolo()
    return YOLO(weights_path)


def _to_rgb_image(arr: np.ndarray) -> Image.Image:
    """
    Ultralytics results.plot() 常回傳 BGR ndarray；這裡轉成 RGB PIL Image 方便顯示。
    """
    if arr.ndim != 3 or arr.shape[2] != 3:
        return Image.fromarray(arr)
    rgb = arr[:, :, ::-1]
    return Image.fromarray(rgb)


def main():
    st.set_page_config(page_title="TireCheckwithYOLO", layout="wide")
    st.title("TireCheckwithYOLO")
    st.caption("Upload an image and run YOLOv8 inference to classify tire status (detection).")

    repo_root = Path(__file__).resolve().parent
    default_weights = repo_root / "runs" / "detect" / "train3" / "weights" / "best.pt"

    with st.sidebar:
        st.header("Settings")
        weights = st.text_input("Weights path (.pt)", value=str(default_weights))
        conf = st.slider("Confidence threshold", min_value=0.01, max_value=0.99, value=0.25, step=0.01)
        imgsz = st.selectbox("Image size (imgsz)", options=[640, 512, 416, 320, 768, 960, 1280], index=0)
        device = st.text_input("Device (optional)", value="", help='e.g. "0" or "cpu"')
        run_btn = st.button("Run inference", type="primary", use_container_width=True)

    uploaded = st.file_uploader("Upload a tire image (jpg/png)", type=["jpg", "jpeg", "png"])

    if not uploaded:
        st.info("請先上傳一張圖片再進行辨識。")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"讀取圖片失敗：{e}")
        return

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    if not run_btn:
        with c2:
            st.subheader("Result")
            st.info("調整左側參數後按「Run inference」。")
        return

    weights_path = Path(weights)
    if not weights_path.exists():
        st.error(f"找不到權重檔：`{weights_path}`")
        return

    model = _load_model(str(weights_path))

    with st.spinner("Running inference..."):
        results = model.predict(
            source=image,
            conf=float(conf),
            imgsz=int(imgsz),
            device=(device.strip() or None),
            save=False,
            verbose=False,
        )

    if not results:
        with c2:
            st.subheader("Result")
            st.warning("沒有返回結果（results 為空）。")
        return

    r0 = results[0]
    plotted = r0.plot()  # ndarray (BGR)
    out_img = _to_rgb_image(plotted)

    with c2:
        st.subheader("Result (Annotated)")
        st.image(out_img, use_container_width=True)

        boxes = getattr(r0, "boxes", None)
        if boxes is None or boxes.cls is None or boxes.conf is None:
            st.warning("沒有偵測到任何框。")
            return

        cls = boxes.cls.cpu().numpy().astype(int).tolist()
        confs = boxes.conf.cpu().numpy().tolist()

        names = getattr(r0, "names", None) or getattr(model, "names", None) or {}
        labels = [names.get(i, str(i)) for i in cls]

        st.markdown("**Detections**")
        st.dataframe(
            {
                "label": labels,
                "confidence": [round(float(c), 4) for c in confs],
            },
            use_container_width=True,
        )

        # Download annotated image
        buf = io.BytesIO()
        out_img.save(buf, format="JPEG", quality=95)
        st.download_button(
            "Download annotated image",
            data=buf.getvalue(),
            file_name="tire_result.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

