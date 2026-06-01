"""CLI 入口模块：argparse 命令行接口函数。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_shot_splitter import pipeline, report
from video_shot_splitter.resources import default_weights_path


def run_transnet_cli(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Split a video into shots with TransNetV2.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--score-mode", choices=["single", "many", "max", "mean"], default="single")
    parser.add_argument("--min-shot-seconds", type=float, default=0.0)
    parser.add_argument("--transition-low-threshold", type=float, default=None)
    parser.add_argument("--transition-peak-threshold", type=float, default=None)
    parser.add_argument("--transition-merge-gap", type=int, default=2)
    parser.add_argument(
        "--use-transition-boundaries",
        action="store_true",
        help="Build shot boundaries from transition peak frames instead of threshold crossings.",
    )
    parser.add_argument("--visualize-report", action="store_true", help="Write an HTML report with timeline, scores, and thumbnails.")
    parser.add_argument("--export-clips", action="store_true", help="Export each detected shot as an mp4 clip.")
    parser.add_argument("--clip-mode", choices=["copy", "reencode"], default="copy")
    args = parser.parse_args(argv)

    weights = args.weights if args.weights is not None else default_weights_path()

    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if not weights.exists():
        raise FileNotFoundError(weights)

    import torch
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Reading video: {args.video}")
    frames, fps, width, height = pipeline.read_video_frames(args.video)
    print(f"Decoded {len(frames)} frames, fps={fps:.3f}, source={width}x{height}")
    print(f"Running TransNetV2 on {device}")
    single, many = pipeline.predict_transnet(frames, weights, device)
    scores = pipeline.select_scores(single, many, args.score_mode)
    transition_low = args.transition_low_threshold if args.transition_low_threshold is not None else args.threshold
    transition_peak = args.transition_peak_threshold if args.transition_peak_threshold is not None else args.threshold
    transitions = pipeline.detect_transition_regions(
        scores,
        single,
        many,
        low_threshold=transition_low,
        peak_threshold=transition_peak,
        merge_gap=args.transition_merge_gap,
    )
    if args.use_transition_boundaries:
        scenes = pipeline.scenes_from_transitions(transitions, len(frames))
    else:
        scenes = pipeline.predictions_to_scenes(scores, args.threshold)
    scenes = pipeline.merge_short_scenes(scenes, int(round(args.min_shot_seconds * fps)))
    csv_path, json_path, pred_path, rows = pipeline.write_transnet_outputs(
        args.output_dir,
        args.video,
        scenes,
        single,
        many,
        fps,
        width,
        height,
        args.threshold,
        args.score_mode,
        args.min_shot_seconds,
    )
    transition_path = pipeline.write_transitions(args.output_dir, args.video, transitions, fps, transition_low, transition_peak)
    if args.visualize_report:
        report_path = report.write_visual_report(
            args.output_dir,
            args.video,
            rows,
            single,
            many,
            scores,
            fps,
            width,
            height,
            args.threshold,
            args.score_mode,
            args.min_shot_seconds,
            transitions=transitions,
        )
        print(f"Wrote report: {report_path}")
    if args.export_clips:
        clips_dir = args.output_dir / f"{args.video.stem}_clips"
        pipeline.export_clips(args.video, rows, clips_dir, ffmpeg_exe=None, clip_mode=args.clip_mode)
        print(f"Wrote clips: {clips_dir}")
    print(f"Detected {len(scenes)} shots")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {pred_path}")
    print(f"Wrote: {transition_path}")


def run_autoshot_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run shotsplit/AutoShot and write frame-level candidate scores.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_autoshot"))
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--transition-pad-frames", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export-clips", action="store_true", help="Accepted for CLI compatibility; clips are generated by the sweep stage.")
    args = parser.parse_args(argv)

    from shotsplit import ShotSplitter
    import numpy as np

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ShotSplitter(device=args.device) as splitter:
        analysis = splitter.analyze(
            args.video,
            threshold=args.threshold,
            include_transition_frames=False,
            include_scores=True,
        )

    _frame_count, _smalls, fps = pipeline.read_video(args.video)
    scores = np.asarray(analysis["frame_scores"], dtype=np.float32)
    predictions_path = args.output_dir / f"{args.video.stem}_autoshot_frame_predictions.csv"
    pipeline.write_autoshot_predictions(predictions_path, scores.tolist(), fps)

    analysis_path = args.output_dir / f"{args.video.stem}_autoshot_analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "transition_pad_frames": args.transition_pad_frames,
                "boundary_count": len(analysis.get("boundaries", [])),
                "boundaries": analysis.get("boundaries", []),
                "segments_excluding_transition_frames": analysis.get("segments", []),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"threshold={args.threshold} boundaries={len(analysis.get('boundaries', []))}")
    print(f"Wrote: {predictions_path}")
    print(f"Wrote: {analysis_path}")


def run_sweep_cli(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="High-recall local candidate sweep for fast transitions.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--transnet-predictions", type=Path, default=None)
    parser.add_argument("--autoshot-predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_candidate_sweep"))
    parser.add_argument("--threshold", type=float, default=0.34, help=argparse.SUPPRESS)
    parser.add_argument("--model-keep-threshold", type=float, default=0.30, help=argparse.SUPPRESS)
    parser.add_argument("--visual-keep-threshold", type=float, default=0.58, help=argparse.SUPPRESS)
    parser.add_argument("--weak-visual-keep-threshold", type=float, default=0.46, help=argparse.SUPPRESS)
    parser.add_argument("--min-supports", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--min-gap-seconds", type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument("--clip-seconds", type=float, default=0.9, help=argparse.SUPPRESS)
    parser.add_argument("--split-runs", action="store_true")
    parser.add_argument("--transition-seconds", type=float, default=0.45, help=argparse.SUPPRESS)
    parser.add_argument("--adaptive-runs", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--hard-pad-frames", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--complex-expand-threshold", type=float, default=0.45, help=argparse.SUPPRESS)
    parser.add_argument("--flash-expand-threshold", type=float, default=0.45, help=argparse.SUPPRESS)
    parser.add_argument("--max-complex-seconds", type=float, default=0.75, help=argparse.SUPPRESS)
    parser.add_argument("--dark-threshold", type=float, default=0.08, help=argparse.SUPPRESS)
    parser.add_argument("--bright-threshold", type=float, default=0.96, help=argparse.SUPPRESS)
    parser.add_argument("--luma-pad-frames", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--min-normal-seconds", type=float, default=0.45, help=argparse.SUPPRESS)
    parser.add_argument("--debug", action="store_true", help="Write candidate reports, candidate clips, and debug metadata.")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_count, smalls, fps = pipeline.read_video(args.video)
    transnet = pipeline.load_model_scores(args.transnet_predictions, frame_count)
    autoshot = pipeline.load_model_scores(args.autoshot_predictions, frame_count)
    signals = pipeline.compute_signals(smalls)
    candidates, score, visual = pipeline.build_candidates(
        transnet,
        autoshot,
        signals,
        fps,
        threshold=args.threshold,
        min_gap_frames=max(1, int(round(args.min_gap_seconds * fps))),
        model_keep_threshold=args.model_keep_threshold,
        visual_keep_threshold=args.visual_keep_threshold,
        weak_visual_keep_threshold=args.weak_visual_keep_threshold,
        min_supports=args.min_supports,
    )
    if args.debug:
        clip_rows = pipeline.export_candidate_clips(args.video, args.output_dir, candidates, fps, args.clip_seconds)
        sweep_report, csv_path = report.write_outputs(args.output_dir, args.video, candidates, clip_rows, fps, frame_count, args.threshold)
    else:
        pipeline.cleanup_debug_outputs(args.output_dir)
    if args.split_runs:
        runs = pipeline.build_runs_from_candidates(
            candidates,
            frame_count,
            fps,
            signals=signals if args.adaptive_runs else None,
            transition_seconds=args.transition_seconds,
            hard_pad_frames=args.hard_pad_frames,
            complex_threshold=args.complex_expand_threshold,
            flash_threshold=args.flash_expand_threshold,
            max_complex_seconds=args.max_complex_seconds,
            brightness=signals["brightness"],
            dark_threshold=args.dark_threshold,
            bright_threshold=args.bright_threshold,
            luma_pad_frames=args.luma_pad_frames,
            min_normal_seconds=args.min_normal_seconds,
        )
        run_rows = pipeline.export_run_clips(args.video, args.output_dir, runs, fps)
        report_path = args.output_dir / "normal_transition_report.html"
        run_csv = pipeline.write_runs_csv(args.output_dir, run_rows) if args.debug else None
        summary = pipeline.build_summary(
            args.output_dir,
            args.video,
            report_path,
            run_rows,
            fps,
            frame_count,
            args.threshold,
            runs_csv_path=run_csv,
        )
        summary_path = pipeline.write_summary_json(args.output_dir, summary)
        run_report = report.write_run_report(args.output_dir, summary, args.video)
        print(f"Wrote: {run_report}")
        print(f"Wrote: {summary_path}")
        if run_csv:
            print(f"Wrote: {run_csv}")
    print(f"candidates={len(candidates)}")
    if args.debug:
        print(f"Wrote: {sweep_report}")
        print(f"Wrote: {csv_path}")


def run_pipeline_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full local shot splitting pipeline.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_pipeline"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--debug", action="store_true", help="Keep intermediate files and candidate debug outputs.")
    args = parser.parse_args(argv)

    video = args.video
    out_dir = args.output_dir
    transnet_dir = out_dir / "transnet"
    autoshot_dir = out_dir / "autoshot"
    result_dir = out_dir / "result"
    for path in [transnet_dir, autoshot_dir, result_dir]:
        path.mkdir(parents=True, exist_ok=True)

    print("[1/3] TransNetV2 candidate scores")
    run_transnet_cli(
        [
            str(video),
            "--device",
            args.device,
            "--threshold",
            "0.1",
            "--score-mode",
            "max",
            "--output-dir",
            str(transnet_dir),
        ]
    )

    print("[2/3] AutoShot candidate scores")
    run_autoshot_cli(
        [
            str(video),
            "--threshold",
            "0.45",
            "--transition-pad-frames",
            "2",
            "--output-dir",
            str(autoshot_dir),
        ]
    )

    print("[3/3] Adaptive sweep and clip export")
    run_sweep_cli(
        [
            str(video),
            "--transnet-predictions",
            str(transnet_dir / f"{video.stem}_frame_predictions.csv"),
            "--autoshot-predictions",
            str(autoshot_dir / f"{video.stem}_autoshot_frame_predictions.csv"),
            "--output-dir",
            str(result_dir),
            "--threshold",
            "0.34",
            "--model-keep-threshold",
            "0.36",
            "--visual-keep-threshold",
            "0.66",
            "--weak-visual-keep-threshold",
            "0.55",
            "--min-supports",
            "3",
            "--min-gap-seconds",
            "0.22",
            "--clip-seconds",
            "0.9",
            "--split-runs",
            "--adaptive-runs",
            "--hard-pad-frames",
            "2",
            "--complex-expand-threshold",
            "0.45",
            "--flash-expand-threshold",
            "0.45",
            "--max-complex-seconds",
            "0.75",
            "--dark-threshold",
            "0.08",
            "--bright-threshold",
            "0.96",
            "--luma-pad-frames",
            "2",
            "--min-normal-seconds",
            "0.45",
            *(["--debug"] if args.debug else []),
        ]
    )

    result_report = result_dir / "normal_transition_report.html"
    if not args.debug:
        import shutil
        for path in [transnet_dir, autoshot_dir]:
            if path.exists():
                shutil.rmtree(path)
    print(f"Done. Open: {result_report}")


def run_smoke_test_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight output-contract smoke test.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_smoke"))
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "smoke_input.mp4"
    frame_count = pipeline.write_synthetic_video(video_path)
    fps = 10.0
    runs = [
        {"label": "normal", "start_frame": 0, "end_frame": 9, "source": ""},
        {"label": "transition", "start_frame": 10, "end_frame": 19, "source": "smoke_transition"},
        {"label": "normal", "start_frame": 20, "end_frame": 29, "source": ""},
    ]
    run_rows = pipeline.export_run_clips(video_path, args.output_dir, runs, fps)
    report_path = args.output_dir / "normal_transition_report.html"
    summary = pipeline.build_summary(args.output_dir, video_path, report_path, run_rows, fps, frame_count, threshold=0.34)
    summary_path = pipeline.write_summary_json(args.output_dir, summary)
    run_report = report.write_run_report(args.output_dir, summary, video_path)

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    assert len(data["clips"]) == 3
    assert data["clips"][0]["start_frame"] == 0
    assert data["clips"][-1]["end_frame"] == frame_count - 1
    assert all((args.output_dir / clip["clip_path"]).exists() for clip in data["clips"])
    assert run_report.exists()
    print(f"Smoke test passed: {summary_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Local video shot and transition splitting pipeline.")
    parser.add_argument("stage", choices=["run", "transnet", "autoshot", "sweep", "smoke-test", "serve"], help="Pipeline stage to run.")
    parser.add_argument("stage_args", nargs=argparse.REMAINDER, help="Arguments passed to the selected stage.")
    args = parser.parse_args(argv)

    if args.stage == "run":
        run_pipeline_cli(args.stage_args)
    elif args.stage == "transnet":
        run_transnet_cli(args.stage_args)
    elif args.stage == "autoshot":
        run_autoshot_cli(args.stage_args)
    elif args.stage == "sweep":
        run_sweep_cli(args.stage_args)
    elif args.stage == "smoke-test":
        run_smoke_test_cli(args.stage_args)
    elif args.stage == "serve":
        from video_shot_splitter.web.app import main as serve_main
        serve_main()

