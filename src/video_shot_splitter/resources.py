"""定位打进包内的数据文件（模型权重等）。"""
from importlib.resources import files
from pathlib import Path


def default_weights_path() -> Path:
    """返回包内 TransNetV2 权重文件的真实路径。"""
    resource = files("video_shot_splitter").joinpath(
        "models/transnetv2-pytorch-weights.pth"
    )
    # 普通 wheel 安装下 files() 返回真实磁盘路径，str() 即可得到可用路径。
    # 注意：不支持 zipimport（.egg/zipapp）——那种场景需 importlib.resources.as_file
    # 配合 ExitStack 把资源解到临时文件，但 30MB 权重不适合 zipimport，此处不处理。
    return Path(str(resource))
