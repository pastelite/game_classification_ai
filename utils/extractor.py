import cv2
import os
from pathlib import Path

def extract_frames(video_path: str, output_dir: str, video_name: str|None=None, interval_ms: int = 1000):
    """Extracts frames from a video at a specified interval."""

    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory {output_dir} does not exist. creating...")
        
    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video = cv2.VideoCapture(video_path)

    try:
        # Calculate fps and frame skip
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps * interval_ms / 1000)

        # Extracting
        frame_i = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame_i % frame_skip == 0:
                # vid_name = video_name if video_name else os.path.basename(video_path).split(".")[0]
                vid_name = video_name if video_name else Path(video_path).stem
                filename = f"{vid_name}_{frame_i:04d}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), frame)

            frame_i += 1

        print(f"Extracted {int(frame_i/frame_skip)} frames at {interval_ms}ms interval (originally {frame_i} frames)")

    finally:
        video.release()