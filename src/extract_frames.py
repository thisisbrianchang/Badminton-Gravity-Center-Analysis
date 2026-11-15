"""Extract video frames into an image sequence."""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2


def extract_frames(video_path: Path, output_folder: Path) -> int:
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到影片檔案：{video_path}")

    output_folder.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")

    frame_count = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_path = output_folder / f"frame_{frame_count:04d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"影格儲存失敗：{frame_path}")
            frame_count += 1
    finally:
        capture.release()

    print(f"已擷取 {frame_count} 張影格至：{output_folder}")
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description="將影片拆解為 JPG 影格")
    parser.add_argument("--video", type=Path, default=Path("data/input/class.mp4"), help="輸入影片路徑")
    parser.add_argument("--output", type=Path, default=Path("data/frames"), help="影格輸出資料夾")
    args = parser.parse_args()
    extract_frames(args.video, args.output)


if __name__ == "__main__":
    main()
