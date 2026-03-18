# TireCheckwithYOLO（YOLOv8 輪胎辨識）

以 Ultralytics YOLOv8 進行輪胎影像的**瑕疵/狀態辨識**（偵測任務），並提供可重現的訓練與推論入口。

## 功能概述

- **訓練**：使用自訂資料集 `Tire_dataset/tiredata.yaml` 訓練 YOLOv8 偵測模型
- **推論**：對單張圖片/資料夾做辨識，輸出含框結果圖到 `runs/`
- **類別**：4 類（`Aging`, `frayed`, `craked`, `Normal`）

## 主要入口（建議使用）

- **訓練**：`train.py`
- **辨識/推論**：`predict.py`

## 安裝

```bash
python -m pip install -r requirements.txt
```

> 備註：本專案根目錄有一份 `ultralytics/` 原始碼資料夾，容易遮蔽 pip 套件；`train.py`/`predict.py` 已做處理，會優先使用 pip 安裝的 `ultralytics`。

## 版本控管（重要）

- **不會上傳**：資料集 `Tire_dataset/`、訓練輸出 `runs/`、以及權重檔 `*.pt`
- 已在 `.gitignore` 排除，避免 repo 過大/推不上去

## Git / GitHub（連結與上傳）

> 你已安裝 Git，可用 `git --version` 確認。

### 初始化與第一次提交（在專案根目錄）

```bash
git init
git add .
git commit -m "init TireCheckwithYOLO"
```

### 連結到 GitHub 並推送

1. 到 GitHub 建立空 repo：`TireCheckwithYOLO`
2. 在本機設定遠端並推送：

```bash
git branch -M main
git remote add origin https://github.com/CMYang95/TireCheckwithYOLO.git
git push -u origin main
```

> 因為 `.gitignore` 已排除 `Tire_dataset/`、`runs/`、`*.pt`，所以這些不會被推到 GitHub。

## 辨識（輸出結果）

```bash
python predict.py --weights runs/detect/train3/weights/best.pt --source Tire_dataset/images/tire22.jpg --conf 0.2 --name predict_demo
```

- 輸出資料夾：`runs/detect/.../predict_demo/`
 - 若你本機沒有權重或資料集，請先放置到對應路徑，或用參數改成你的路徑

## 訓練

```bash
python train.py --data Tire_dataset/tiredata.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --name detect/train_tire
```

## 資料集

- `Tire_dataset/tiredata.yaml` 已改成 **相對路徑**（可移動專案資料夾、不再綁死 `D:\TireCheck`）。

