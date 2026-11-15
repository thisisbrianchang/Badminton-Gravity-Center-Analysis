"""Detect and segment people in images with a pretrained Mask R-CNN model."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load pretrained Mask R-CNN weights (COCO). First run may download weights."""
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    model.to(DEVICE)
    model.eval()
    return model


def find_person(
    image_path: Path,
    model,
    score_threshold: float = 0.8,
    min_area: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the person-masked BGR image and a binary person mask."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"無法讀取影像：{image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = to_tensor(image_rgb).to(DEVICE)
    with torch.inference_mode():
        prediction = model([image_tensor])[0]

    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for label, score, predicted_mask in zip(
        prediction["labels"], prediction["scores"], prediction["masks"]
    ):
        # COCO class id 1 represents a person.
        if int(label.item()) != 1 or float(score.item()) <= score_threshold:
            continue
        person_mask = (predicted_mask[0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
        if int(np.count_nonzero(person_mask)) >= min_area:
            combined_mask = cv2.bitwise_or(combined_mask, person_mask)

    masked_image = cv2.bitwise_and(image, image, mask=combined_mask)
    return masked_image, combined_mask


def process_all_frames(
    input_folder: Path,
    output_folder: Path,
    model,
    score_threshold: float = 0.8,
    min_area: int = 500,
) -> int:
    if not input_folder.is_dir():
        raise FileNotFoundError(f"找不到影格資料夾：{input_folder}")

    frame_files = sorted(input_folder.glob("*.jpg"))
    if not frame_files:
        raise ValueError(f"資料夾內沒有 JPG 影格：{input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)
    for frame_path in frame_files:
        masked_image, _ = find_person(
            frame_path, model, score_threshold=score_threshold, min_area=min_area
        )
        output_path = output_folder / frame_path.name
        if not cv2.imwrite(str(output_path), masked_image):
            raise RuntimeError(f"遮罩影像儲存失敗：{output_path}")

    print(f"已處理 {len(frame_files)} 張影格，輸出至：{output_folder}")
    return len(frame_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Mask R-CNN 偵測並分割影格中的人物")
    parser.add_argument("--input", type=Path, default=Path("data/frames"), help="輸入影格資料夾")
    parser.add_argument("--output", type=Path, default=Path("data/masks"), help="遮罩影像輸出資料夾")
    parser.add_argument("--score-threshold", type=float, default=0.8, help="人物偵測信心分數門檻")
    parser.add_argument("--min-area", type=int, default=1200, help="人物遮罩最小像素面積")
    args = parser.parse_args()
    model = load_model()
    process_all_frames(args.input, args.output, model, args.score_threshold, args.min_area)


if __name__ == "__main__":
    main()
