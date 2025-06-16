import decord
from pathlib import Path

video_path = Path("/disk1/jinmin/CogVideo_dataset/360x_dataset_HR/cogvideo_dataset/GT_videos_mp4/019cc67f-512f-4b8a-96ef-81f806c86ce1/Segment_0000/GT_0-90.mp4")
try:
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    print(f"Successfully loaded video with {len(video_reader)} frames")
except Exception as e:
    print(f"Failed to load video: {e}")