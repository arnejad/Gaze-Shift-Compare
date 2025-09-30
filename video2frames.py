#!/usr/bin/env python3
from pathlib import Path
import cv2

# 🔧 Set your root directory here
ROOT_DIR = Path("/media/ash/Expansion/data/drews-dynamic")

def extract_frames(video_path: Path, out_dir: Path) -> int:
    """Extract frames from video_path into out_dir as 000001.png, 000002.png, ..."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 1
    ok, frame = cap.read()
    while ok:
        filename = f"{frame_idx:06d}.png"
        out_path = out_dir / filename
        if not cv2.imwrite(str(out_path), frame):
            cap.release()
            raise RuntimeError(f"Failed to write frame to {out_path}")
        frame_idx += 1
        ok, frame = cap.read()

    cap.release()
    return frame_idx - 1

def main():
    if not ROOT_DIR.is_dir():
        print(f"Error: {ROOT_DIR} is not a directory.")
        return

    total_videos = 0
    total_frames = 0

    for folder in sorted(ROOT_DIR.iterdir()):
        if not folder.is_dir():
            continue

        video = folder / "scene_camera.mp4"
        if not video.exists():
            continue  # skip if no video

        out_dir = folder / "image_2"

        try:
            print(f"Processing: {video}")
            frames = extract_frames(video, out_dir)
            print(f"  -> Saved {frames} frames to {out_dir}")
            total_videos += 1
            total_frames += frames
        except Exception as e:
            print(f"  !! Error in {folder}: {e}")

    print("\n=== Summary ===")
    print(f"Processed videos: {total_videos}")
    print(f"Total frames saved: {total_frames}")

if __name__ == "__main__":
    main()