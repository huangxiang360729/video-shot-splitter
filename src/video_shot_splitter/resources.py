"""定位打进包内的数据文件（模型权重等）。"""
from importlib.resources import files
from pathlib import Path


def default_weights_path() -> Path:
    """返回包内 TransNetV2 权重文件的真实路径。"""
    resource = files("video_shot_splitter").joinpath(
        "models/transnetv2-pytorch-weights.pth"
    )
    # importlib.resources 在普通安装下返回真实路径；用 as_file 兜底 zip 安装
    return Path(str(resource))
