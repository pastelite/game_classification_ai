import cv2
import os
from pathlib import Path
from tqdm import tqdm

# def extract_frames(video_path: str, output_dir: str, video_name: str|None=None, interval_ms: int = 1000):
#     """Extracts frames from a video at a specified interval."""

#     # Make sure output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)
#         print(f"Output directory {output_dir} does not exist. creating...")

#     # Check if video exists
#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"Video file not found: {video_path}")

#     video = cv2.VideoCapture(video_path)

#     try:
#         # Calculate fps and frame skip
#         fps = video.get(cv2.CAP_PROP_FPS)
#         frame_skip = int(fps * interval_ms / 1000)

#         # Extracting
#         frame_i = 0
#         while True:
#             ret, frame = video.read()
#             if not ret:
#                 break

#             if frame_i % frame_skip == 0:
#                 # vid_name = video_name if video_name else os.path.basename(video_path).split(".")[0]
#                 vid_name = video_name if video_name else Path(video_path).stem
#                 filename = f"{vid_name}_{frame_i:04d}.jpg"
#                 cv2.imwrite(os.path.join(output_dir, filename), frame)

#             frame_i += 1

#         print(f"Extracted {int(frame_i/frame_skip)} frames at {interval_ms}ms interval (originally {frame_i} frames)")

#     finally:
#         video.release()


def extract_frames_interval(
    video_path: str, output_dir: str, name: str | None = None, interval=30
) -> tuple[int, int, int]:
    """
    Extract frames from a video at a specified frame interval

    Returns
    -------
    tuple[int,int,int]
        (number of frames extracted, total number of frames, interval)
    """

    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory {output_dir} does not exist. creating...")

    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video = cv2.VideoCapture(video_path)
    vid_name = name if name else Path(video_path).stem

    try:
        # Extracting
        # frame_i = 0
        # while True:
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # num_frames_extract = num_frames // interval
        for frame_i in (progress_bar := tqdm(range(0, num_frames, interval))):
            # skipping frames
            # frame_i = set_i * interval
            progress_bar.set_description(f"{vid_name} [{frame_i}/{num_frames}]")
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_i)

            ret, frame = video.read()
            if not ret:
                break

            if frame_i % interval == 0:
                # vid_name = video_name if video_name else os.path.basename(video_path).split(".")[0]
                filename = f"{vid_name}_{frame_i:04d}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), frame)
                # progress_bar.set_postfix_str(f"n={int(frame_i/interval)+1}")

        # print(f"Extracted {int(frame_i/interval)} frames at {interval} frames interval (originally {frame_i} frames)")

    finally:
        video.release()
        return int(frame_i / interval) + 1, num_frames, interval


def extract_n_frames(
    video_path: str, output_dir: str, video_name: str | None = None, n: int = 0
):
    """
    Extract frame_n frames from a video

    Returns
    -------
    tuple[int,int,int]
        (number of frames extracted, total number of frames, interval)
    """
    if n <= 0:
        return 0, 0, 0

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video = cv2.VideoCapture(video_path)
    interval_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) // (
        n - 1
    )  # n-1 because we start from 0

    return extract_frames_interval(video_path, output_dir, video_name, interval_frame)


def extract_frames_time_interval(
    video_path: str,
    output_dir: str,
    video_name: str | None = None,
    interval: int = 1000,
):
    """
    Extract frames from a video at a specified time interval

    Parameters
    ----------
    video_path : str
        Path to the video file
    output_dir : str
        Path to the output directory
    video_name : str, optional
        Name of the video that will appended to filename, defaulting to the video file name
    interval : int, optional
        Time interval in milliseconds, defaulting to 1000 (1 second)

    Returns
    -------
    tuple[int,int,int]
        (number of frames extracted, total number of frames, interval)
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    interval_frame = int(fps * interval / 1000)

    return extract_frames_interval(video_path, output_dir, video_name, interval_frame)
