# 羽球選手人體分割與重心軌跡分析

**Python × OpenCV × PyTorch × Mask R-CNN**

本專案以影片影格為輸入，使用預訓練 Mask R-CNN 偵測並分割影像中的人物，再透過 OpenCV `moments` 計算人物二值遮罩的 2D 幾何中心，最後將中心點與移動軌跡繪製回影片。

> **重要名詞說明：** 目前程式計算的是人物分割遮罩在影像平面上的「幾何中心」，可作為觀察人物位置變化的近似指標；並非經過生物力學量測的真實人體質心。

## 專案功能

1. **影片影格擷取：** 使用 OpenCV 將影片逐格擷取為 JPG。
2. **人體偵測與分割：** 使用 COCO 預訓練 Mask R-CNN（ResNet-50-FPN）辨識人物並產生遮罩。
3. **中心座標計算：** 使用 `cv2.moments()` 計算人物遮罩的影像幾何中心。
4. **移動軌跡視覺化：** 將中心點與累積軌跡繪製到輸出影片。
5. **CPU／GPU 支援：** 若 PyTorch 偵測到可用 CUDA，會優先使用 GPU，否則使用 CPU。

## 使用技術

- Python
- OpenCV
- PyTorch、Torchvision
- Mask R-CNN ResNet-50-FPN（COCO 預訓練權重）
- NumPy
- Image Moments、二值遮罩、2D 幾何中心計算

## 專案架構

```text
Badminton-Gravity-Center-Analysis/
├── src/
│   ├── extract_frames.py   # 影片影格擷取
│   ├── find_person.py      # 人體偵測與分割
│   └── process_video.py    # 中心計算與軌跡影片輸出
├── data/
│   └── input/              # 放置自行準備的輸入影片（不納入 Git）
├── outputs/                # 輸出影片（不納入 Git）
├── .gitignore
├── README.md
├── README.zh-TW.md
└── requirements.txt
```

## 安裝方式

### 1. 建立虛擬環境（Windows PowerShell）

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. 安裝 PyTorch 與 Torchvision

請依照作業系統及 CUDA 版本，使用官方安裝選擇器取得相容的 PyTorch／Torchvision 安裝指令：

https://pytorch.org/get-started/locally/

兩者版本必須相容。完成後安裝其他套件：

```powershell
pip install -r requirements.txt
```

第一次執行模型時，Torchvision 可能會下載預訓練權重；若本機尚未快取權重，需要網路連線。

## 執行方式

請先將自己有權使用的影片放入 `data/input/`，例如 `class.mp4`。原始影片不會隨公開 Repository 提供。

請在專案根目錄執行以下指令。

### 步驟一：擷取影片影格

```powershell
python src/extract_frames.py --video data/input/class.mp4 --output data/frames
```

### 步驟二：偵測並分割人物（可選的中間結果）

```powershell
python src/find_person.py --input data/frames --output data/masks --min-area 1200 --score-threshold 0.8
```

分割後的影像會儲存在 `data/masks/`。

### 步驟三：計算中心並輸出軌跡影片

```powershell
python src/process_video.py --frames data/frames --output outputs/gravity_center_analysis.mp4 --min-area 1200 --score-threshold 0.8 --fps 30
```

`--offset-y` 可調整軌跡線的垂直顯示位置，預設為 `0`。

## 參數說明

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `--score-threshold` | `0.8` | 人體偵測信心分數門檻 |
| `--min-area` | 影片流程為 `1200` | 遮罩最小像素面積 |
| `--fps` | `30` | 輸出影片影格率 |
| `--offset-y` | `0` | 軌跡線垂直顯示位移 |

## 目前限制

- 程式使用 COCO 的 person 類別（label `1`）。同一影格中的多個人物遮罩會合併，因此多人同時出現時，計算結果是合併遮罩的幾何中心，並非個別選手的追蹤結果。
- 幾何中心可能受到攝影機視角、遮擋、分割品質及人物在畫面中的比例影響。
- 目前尚未實作個體追蹤、攝影機校正、3D 重建或生物力學真實質心估測。
- 執行速度會依影片解析度、影格數、CPU／GPU 硬體及模型設定而不同。
- 請勿將含有未經授權、可識別個人或私人內容的影片、影格、模型權重及個人資料上傳至公開 Repository。

## 專案分工

此 Repository 整理影片影格擷取、人體分割、遮罩幾何中心計算及移動軌跡視覺化的 Python 程式。正式作為個人作品集前，請依實際情況補上個人負責項目與專案背景。

## 授權

目前未加入 License。未指定授權前，其他人不應假設可以重用、修改或散布本專案程式碼。
