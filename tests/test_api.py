def test_run_pipeline_importable():
    from video_shot_splitter import run_pipeline, __version__
    assert callable(run_pipeline)
    assert isinstance(__version__, str)


def test_write_synthetic_video_returns_frame_count(tmp_path):
    from video_shot_splitter.pipeline import write_synthetic_video
    video_path = tmp_path / "smoke.mp4"
    frame_count = write_synthetic_video(video_path)
    assert isinstance(frame_count, int), f"expected int, got {type(frame_count)}"
    assert frame_count == 30  # 3 colors x 10 frames each
