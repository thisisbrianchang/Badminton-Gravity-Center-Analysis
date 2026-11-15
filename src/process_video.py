"""Calculate person-mask centroids and render their trajectory on a video."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from find_person import find_person, load_model


def calculate_gravity_center(mask: np.ndarray) -> tuple[int, int]:
    """Calculate the binary mask's image centroid using image moments.

    This is the 2D centroid of the detected silhouette, not a biomechanical
    measurement of the body's true center of mass.
    """
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return -1, -1
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    return center_x, center_y


def frames_to_video(
    frame_folder: Path,
    output_video: Path,
    gravity_centers: list[tuple[int, int]],
    fps: float = 30.0,
    offset_y: int = 0,
) -> None:
    frame_files = sorted(frame_folder.glob("*.jpg"))
    if not frame_files:
        raise ValueError(f"找不到 JPG 影格：{frame_folder}")
    if len(frame_files) != len(gravity_centers):
        raise ValueError("影格數量與重心座標數量不一致。")

    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        raise ValueError(f"無法讀取影格：{frame_files[0]}")
    height, width = first_frame.shape[:2]

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"無法建立輸出影片：{output_video}")

    trajectory: list[tuple[int, int]] = []
    try:
        for frame_path, (center_x, center_y) in zip(frame_files, gravity_centers):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"無法讀取影格：{frame_path}")

            if center_x >= 0 and center_y >= 0:
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                trajectory.append((center_x, center_y - offset_y))

            for index in range(1, len(trajectory)):
                cv2.line(frame, trajectory[index - 1], trajectory[index], (0, 255, 0), 2)
            writer.write(frame)
    finally:
        writer.release()

    print(f"已輸出分析影片：{output_video}")


def main() -> None:
    parser = argparse.ArgumentParser(description="計算人物遮罩中心並輸出移動軌跡影片")
    parser.add_argument("--frames", type=Path, default=Path("data/frames"), help="輸入影格資料夾")
    parser.add_argument("--output", type=Path, default=Path("outputs/gravity_center_analysis.mp4"), help="輸出影片路徑")
    parser.add_argument("--score-threshold", type=float, default=0.8, help="人物偵測信心分數門檻")
    parser.add_argument("--min-area", type=int, default=1200, help="人物遮罩最小像素面積")
    parser.add_argument("--fps", type=float, default=30.0, help="輸出影片影格率")
    parser.add_argument("--offset-y", type=int, default=0, help="軌跡線垂直位移量")
    args = parser.parse_args()

    frame_files = sorted(args.frames.glob("*.jpg"))
    if not frame_files:
        raise ValueError(f"資料夾內沒有 JPG 影格：{args.frames}")

    model = load_model()
    centers: list[tuple[int, int]] = []
    for frame_path in frame_files:
        _, mask = find_person(
            frame_path, model,
            score_threshold=args.score_threshold,
            min_area=args.min_area,
        )
        centers.append(calculate_gravity_center(mask))

    frames_to_video(args.frames, args.output, centers, fps=args.fps, offset_y=args.offset_y)


if __name__ == "__main__":
    main()
