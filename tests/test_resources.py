from pathlib import Path
from video_shot_splitter.resources import default_weights_path


def test_default_weights_path_exists():
    p = default_weights_path()
    assert isinstance(p, Path)
    assert p.name == "transnetv2-pytorch-weights.pth"
    assert p.exists()
