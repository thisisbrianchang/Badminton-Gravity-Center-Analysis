# Badminton Player Segmentation & Gravity-Center Trajectory Analysis

**羽球選手人體分割與重心軌跡分析｜Python × OpenCV × PyTorch × Mask R-CNN**

本專案以影片影格為輸入，使用預訓練 Mask R-CNN 偵測並分割影像中的人物，再透過 OpenCV image moments 計算人物二值遮罩的 2D 幾何中心，最後將中心點與移動軌跡繪製回影片。

> **研究用語說明：** 程式計算的是人物分割遮罩在影像平面上的幾何中心（2D silhouette centroid），可作為觀察人物位置變化的近似指標；它不是經過生物力學量測的真實人體質心（center of mass）。

## Features

- **Video frame extraction**：使用 OpenCV 將影片逐格擷取為 JPG。
- **Person segmentation**：使用 COCO 預訓練 Mask R-CNN（ResNet-50-FPN）辨識人物並產生遮罩。
- **Centroid calculation**：使用 `cv2.moments()` 計算人物遮罩的影像幾何中心。
- **Trajectory visualization**：將中心點及累積移動軌跡繪製至輸出影片。
- **CPU / CUDA**：若 PyTorch 可使用 CUDA，模型會優先使用 GPU，否則使用 CPU。

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Computer vision | OpenCV |
| Deep learning | PyTorch, Torchvision |
| Model | Mask R-CNN ResNet-50-FPN (COCO pretrained weights) |
| Numerical processing | NumPy |
| Image analysis | Binary masks, image moments, 2D centroid |

## Project Structure

```text
Badminton-Gravity-Center-Analysis/
├── src/
│   ├── extract_frames.py   # Extract video frames
│   ├── find_person.py      # Person detection and segmentation
│   └── process_video.py    # Centroid calculation and trajectory rendering
├── data/
│   └── input/              # Put your own source video here (not tracked by Git)
├── outputs/                # Generated videos (not tracked by Git)
├── .gitignore
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.9 or later (a compatible Python / PyTorch combination is required)
- OpenCV
- NumPy
- PyTorch and Torchvision with matching compatible versions

### 1. Create a virtual environment

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install PyTorch and Torchvision

Choose the correct command for your operating system and CUDA version from the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

Install a compatible pair of `torch` and `torchvision` versions. Then install the remaining packages:

```powershell
pip install -r requirements.txt
```

The first model run may download pretrained Mask R-CNN weights from Torchvision. An internet connection is required for this initial download unless the weights are already cached locally.

## How to Run

Place a video file in `data/input/`, for example `class.mp4`. The input video is intentionally excluded from this public repository; use a video you have permission to process.

Run commands from the repository root.

### Step 1. Extract frames

```powershell
python src/extract_frames.py --video data/input/class.mp4 --output data/frames
```

### Step 2. Detect and segment people (optional intermediate output)

```powershell
python src/find_person.py --input data/frames --output data/masks --min-area 1200 --score-threshold 0.8
```

This command saves person-masked images to `data/masks/`.

### Step 3. Calculate centroids and render the trajectory video

```powershell
python src/process_video.py --frames data/frames --output outputs/gravity_center_analysis.mp4 --min-area 1200 --score-threshold 0.8 --fps 30
```

Optional `--offset-y` shifts the trajectory line vertically for visualization. The default is `0`.

## Main Processing Flow

```text
Input video
    │
    ▼
OpenCV frame extraction
    │
    ▼
JPG image frames
    │
    ▼
Mask R-CNN person segmentation
    │
    ▼
Binary person mask
    │
    ▼
OpenCV image moments
    │
    ▼
2D silhouette centroid per frame
    │
    ▼
Draw centroid points and trajectory
    │
    ▼
Output analysis video
```

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `--score-threshold` | `0.8` | Minimum person detection confidence |
| `--min-area` | `1200` in video workflow | Minimum mask area in pixels |
| `--fps` | `30` | Output video frame rate |
| `--offset-y` | `0` | Vertical display offset for the trajectory line |

## Notes and Limitations

- The model uses the COCO `person` class (label `1`). Multiple detected person masks are combined into one mask; if multiple people appear in a frame, the resulting centroid represents the combined mask, not an individually tracked athlete.
- The calculated centroid is a 2D image-mask centroid. Camera perspective, occlusion, segmentation quality and subject scale can affect its position.
- The current workflow does not perform identity tracking, camera calibration, 3D reconstruction, or biomechanical center-of-mass estimation.
- Processing speed depends on video resolution, frame count, CPU/GPU hardware and model configuration.
- Do not commit private, identifiable, or unlicensed video footage, generated frames, model weights, or personal data to a public repository.

## Contribution / Project Work

This repository organizes the Python implementation for video frame extraction, person segmentation, mask-based centroid calculation and trajectory visualization. Update this section with your exact individual responsibilities and project context before using it as a formal portfolio statement.

## License

No license has been added. Unless a license is selected, others should not assume they have permission to reuse, modify, or redistribute this code.
