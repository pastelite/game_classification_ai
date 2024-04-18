from pytube import YouTube
from pytube.cli import on_progress
import os.path as path
import os
import warnings
import pathlib
import unittest


def download_youtube_video(
    link: str, output_path="./download", no_create_dir=False, show_progress=False
):
    if show_progress:
        print(f"Downloading {link} to {output_path}")
    yt = YouTube(link, on_progress_callback=on_progress if show_progress else None)
    path = pathlib.Path(output_path)

    if path.exists() and path.is_file():
        print(f"Output file {output_path} is already exists, skipping...")
        return

    if path.is_dir():
        if not path.exists():
            if no_create_dir:
                raise FileNotFoundError(
                    f"Output directory {output_path} does not exist."
                )
            # warnings.warn(f"Output directory {output_path} does not exist. creating...")
            path.mkdir(parents=True)
        filename = path / (yt.video_id + ".mp4")
    else:
        if not path.parent.exists():
            if no_create_dir:
                raise FileNotFoundError(
                    f"Output directory {output_path} does not exist."
                )
            # warnings.warn(f"Output directory {path.parent} does not exist. creating...")
            path.parent.mkdir(parents=True)
        filename = path
        if path.suffix == "":
            filename = path.with_suffix(".mp4")

    stream = yt.streams.filter(
        progressive=True, file_extension="mp4"
    ).get_lowest_resolution()
    if stream:
        stream.download(filename=str(filename))
    else:
        warnings.warn(f"The video {yt.title} has no stream found")


class TestDownloadYoutubeVideo(unittest.TestCase):
    def test_download_no_ext(self):
        os.makedirs("./download_test", exist_ok=True)
        download_youtube_video(
            "https://www.youtube.com/watch?v=fxqE27gIZcc", "./download_test/test"
        )
        self.assertTrue(path.exists("./download_test/test.mp4"))
        os.remove("./download_test/test.mp4")
        os.rmdir("./download_test")

    def test_download_ext(self):
        os.makedirs("./download_test", exist_ok=True)
        download_youtube_video(
            "https://www.youtube.com/watch?v=fxqE27gIZcc", "./download_test/test.mp4"
        )
        self.assertTrue(path.exists("./download_test/test.mp4"))
        os.remove("./download_test/test.mp4")
        os.rmdir("./download_test")

    def test_download_auto_path(self):
        with self.assertWarns(UserWarning):
            download_youtube_video(
                "https://www.youtube.com/watch?v=fxqE27gIZcc",
                "./download_test/test.mp4",
            )
        self.assertTrue(path.exists("./download_test/test.mp4"))
        os.remove("./download_test/test.mp4")
        os.rmdir("./download_test")

    def test_download_no_create_dir(self):
        with self.assertRaises(FileNotFoundError):
            download_youtube_video(
                "https://www.youtube.com/watch?v=fxqE27gIZcc",
                "./download_test_not_exists/test",
                no_create_dir=True,
            )

    def test_download_youtudotbe_only(self):
        os.makedirs("./download_test", exist_ok=True)
        download_youtube_video("youtu.be/fxqE27gIZcc", "./download_test/test")
        self.assertTrue(path.exists("./download_test/test.mp4"))
        os.remove("./download_test/test.mp4")
        os.rmdir("./download_test")


if __name__ == "__main__":
    unittest.main()
