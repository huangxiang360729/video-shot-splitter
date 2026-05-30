import argparse
import csv
import json
from pathlib import Path

import numpy as np
from shotsplit import ShotSplitter

from visualize_frame_transitions import (
    build_runs,
    compute_visual_change,
    export_run_clips,
    format_time,
    read_video,
    save_storyboard,
    write_clip_manifest,
    write_csvs,
    write_html_report,
)


def write_autoshot_predictions(path: Path, scores: list[float], fps: float):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "seconds", "single_frame_score", "many_hot_score"])
        for idx, score in enumerate(scores):
            writer.writerow([idx, round(idx / fps, 3), float(score), float(score)])


def labels_from_boundaries(boundaries: list[dict], frame_count: int, pad_frames: int):
    labels = np.zeros(frame_count, dtype=np.uint8)
    transitions = []
    for boundary in boundaries:
        start = max(0, int(boundary["run_start_frame"]) - pad_frames)
        end = min(frame_count - 1, int(boundary["run_end_frame"]) + pad_frames)
        peak = int(boundary["peak_frame"])
        labels[start : end + 1] = 1
        transitions.append(
            {
                "transition_id": len(transitions) + 1,
                "start_frame": start,
                "end_frame": end,
                "peak_frame": peak,
                "duration_frames": end - start + 1,
                "peak_fused_score": round(float(boundary["peak_score"]), 6),
                "peak_model_score": round(float(boundary["peak_score"]), 6),
                "peak_many_hot_score": round(float(boundary["peak_score"]), 6),
                "peak_visual_score": "",
                "region_model_peak": round(float(boundary["peak_score"]), 6),
                "region_many_hot_peak": round(float(boundary["peak_score"]), 6),
                "region_visual_peak": "",
                "prepost_distance": "",
                "motion_residual": "",
                "motion_inlier_ratio": "",
                "transition_type": "boundary_run",
            }
        )
    return labels, transitions


def main():
    parser = argparse.ArgumentParser(description="Run shotsplit/AutoShot and build clip wall report.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_autoshot"))
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--transition-pad-frames", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export-clips", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem

    with ShotSplitter(device=args.device) as splitter:
        analysis = splitter.analyze(
            args.video,
            threshold=args.threshold,
            include_transition_frames=False,
            include_scores=True,
        )

    frames, thumbs, smalls, fps, source_width, source_height = read_video(args.video)
    frame_count = len(frames)
    scores = np.asarray(analysis["frame_scores"], dtype=np.float32)[:frame_count]
    labels, transitions = labels_from_boundaries(analysis["boundaries"], frame_count, args.transition_pad_frames)
    runs = build_runs(labels)
    visual = compute_visual_change(smalls)["visual_score"]

    predictions_path = args.output_dir / f"{stem}_autoshot_frame_predictions.csv"
    write_autoshot_predictions(predictions_path, scores.tolist(), fps)

    labels_path, transitions_path, runs_path = write_csvs(
        args.output_dir,
        stem,
        labels,
        transitions,
        runs,
        fps,
        scores,
        scores,
        visual,
        scores,
    )

    storyboard_path = args.output_dir / f"{stem}_collapsed_storyboard.jpg"
    save_storyboard(storyboard_path, thumbs, runs, fps, scores, max_transition_frames=24)

    clips_path = None
    clip_rows = None
    if args.export_clips:
        _, clip_rows = export_run_clips(args.video, args.output_dir, stem, runs, fps)
        clips_path = write_clip_manifest(args.output_dir, stem, clip_rows)

    stats = {
        "frame_count": frame_count,
        "transition_frame_count": int(labels.sum()),
        "transition_count": len(transitions),
        "normal_run_count": sum(1 for run in runs if run["label"] == "normal"),
        "fps": fps,
        "source_width": source_width,
        "source_height": source_height,
    }
    (args.output_dir / f"{stem}_autoshot_analysis.json").write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "transition_pad_frames": args.transition_pad_frames,
                "stats": stats,
                "boundaries": analysis["boundaries"],
                "segments_excluding_transition_frames": analysis["segments"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = write_html_report(
        args.output_dir,
        stem,
        args.video,
        storyboard_path,
        labels_path,
        transitions_path,
        runs_path,
        stats,
        clips_path=clips_path,
        clip_rows=clip_rows,
    )

    print(f"threshold={args.threshold} boundaries={len(analysis['boundaries'])} clips={len(runs)}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {transitions_path}")
    print(f"Wrote: {runs_path}")
    print(f"Wrote: {predictions_path}")
    if clips_path:
        print(f"Wrote: {clips_path}")


if __name__ == "__main__":
    main()
