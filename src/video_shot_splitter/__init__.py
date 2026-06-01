"""video-shot-splitter: 本地视频分镜/转场拆分工具。"""

__version__ = "0.1.0"

from video_shot_splitter.pipeline import run_pipeline

__all__ = ["run_pipeline", "__version__"]
