import sys
import warnings
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.downloader import download_youtube_video

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

def extract_frames(
    video_path: str, output_dir: str, video_name: str | None = None, frame_list: list[int] = [], auto_create_dir: bool = True
) -> int:
    # create output directory if not exists
    if not os.path.exists(output_dir):
        if auto_create_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            raise FileNotFoundError(f"Output directory {output_dir} does not exist.")
        
    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video
    video = cv2.VideoCapture(video_path)
    vid_name = video_name if video_name else Path(video_path).stem
    sorted_frame_list = sorted(frame_list)
    num_extracted = 0
    
    try:
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_i in (progress_bar := tqdm(sorted_frame_list)):
            progress_bar.set_description(f"{vid_name} [{frame_i}/{num_frames}]")
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
            
            ret, frame = video.read()
            if not ret:
                warnings.warn(f"Frame {frame_i} not found in the video {video_path}")
                continue
            
            filename = f"{vid_name}_{frame_i:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            num_extracted += 1

    finally:
        video.release()
        return num_extracted

def extract_frames_interval(
    video_path: str, output_dir: str, name: str | None = None, interval=30.0, auto_create_dir: bool = True
) -> int:
    """
    Extract frames from a video at a specified frame interval

    Returns
    -------
    int
        Number of frames extracted
    """
    
    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # get video length
    video = cv2.VideoCapture(video_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = np.round(np.arange(0, num_frames, interval), 0).astype(int).tolist()
    return extract_frames(video_path, output_dir, name, frames, auto_create_dir)

def extract_frames_n(
    video_path: str, output_dir: str, video_name: str | None = None, n: int = 0
) -> int:
    """
    Extract frame_n frames from a video

    Returns
    -------
    int
        Number of frames extracted
    """
    
    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # get video length
    video = cv2.VideoCapture(video_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = np.round(np.linspace(0, num_frames-1, n), 0).astype(int).tolist()
    return extract_frames(video_path, output_dir, video_name, frames)

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
    tuple[int,float]
        (number of frames extracted, interval)
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    interval_frame = fps * interval / 1000
    
    frames = np.round(np.arange(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)), interval_frame),0).astype(int).tolist()
    return extract_frames(video_path, output_dir, video_name, frames), interval_frame

import unittest

class TestExtractor(unittest.TestCase):
    video_path = "test_temp/bbb.mp4"
    output_dir = "test_temp/output"
    
    @classmethod
    def setUpClass(cls):
        # make sure test directory exists
        if not os.path.exists("test_temp"):
            os.makedirs("test_temp", exist_ok=True)
        if not os.path.exists("test_temp/output"):
            os.makedirs("test_temp/output", exist_ok=True)
            
        # download video if not exists
        video_path = "test_temp/bbb.mp4"
        if not os.path.exists(video_path):
            download_youtube_video("https://www.youtube.com/watch?v=YE7VzlLtp-4", video_path)
    
    @classmethod
    def tearDown(cls):
        # remove test directory
        if os.path.exists("test_temp/output"):
            files = os.listdir("test_temp/output")
            for file in files:
                os.remove(os.path.join("test_temp/output", file))
    
    def test_extract_frames(self):
        frame_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        
        # download video if not exists
        if not os.path.exists(self.video_path):
            download_youtube_video("https://www.youtube.com/watch?v=YE7VzlLtp-4",self.video_path)
            
        num_frames = extract_frames(self.video_path, self.output_dir, frame_list=frame_list)
        self.assertEqual(num_frames, len(frame_list))
        
    def test_extract_frames_interval(self):
        # download video if not exists
        if not os.path.exists(self.video_path):
            download_youtube_video("https://www.youtube.com/watch?v=YE7VzlLtp-4", self.video_path)
            
        num_frames = extract_frames_interval(self.video_path, self.output_dir, interval=240)
        self.assertEqual(num_frames, 60)
        
    def test_extract_frames_n(self):
        # download video if not exists
        if not os.path.exists(self.video_path):
            download_youtube_video("https://www.youtube.com/watch?v=YE7VzlLtp-4", self.video_path)
            
        num_frames = extract_frames_n(self.video_path, self.output_dir, n=10)
        self.assertEqual(num_frames, 10)
        
    def test_extract_frames_time_interval(self):
        # download video if not exists
        if not os.path.exists(self.video_path):
            download_youtube_video("https://www.youtube.com/watch?v=YE7VzlLtp-4", self.video_path)
            
        num_frames, frame_interval = extract_frames_time_interval(self.video_path, self.output_dir, interval=10000)
        self.assertEqual(num_frames, 60)
        self.assertEqual(frame_interval, 240.0)

        

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    unittest.main()