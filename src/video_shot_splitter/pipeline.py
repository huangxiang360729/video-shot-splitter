"""分析/计算管道模块：所有分析与计算函数。"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from video_shot_splitter import report


# ---------------------------------------------------------------------------
# TransNetV2 vendor path setup
# ---------------------------------------------------------------------------

def _get_transnet_dir() -> Path:
    """返回包内 vendor/transnetv2 目录的路径。"""
    try:
        from importlib.resources import files
        resource = files("video_shot_splitter").joinpath("vendor/transnetv2")
        return Path(str(resource))
    except Exception:
        return Path(__file__).resolve().parent / "vendor" / "transnetv2"


_TRANSNET_DIR = _get_transnet_dir()
if str(_TRANSNET_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSNET_DIR))

try:
    from transnetv2_pytorch import TransNetV2  # noqa: E402
except Exception:
    TransNetV2 = None


# ---------------------------------------------------------------------------
# Core utility functions
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def read_video_frames(video_path: Path) -> tuple[np.ndarray, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_small = cv2.resize(frame_rgb, (48, 27), interpolation=cv2.INTER_AREA)
        frames.append(frame_small)

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    if not fps or fps <= 0:
        fps = 25.0

    return np.asarray(frames, dtype=np.uint8), float(fps), width, height


def iter_transnet_windows(frames: np.ndarray):
    no_padded_frames_start = 25
    remainder = len(frames) % 50
    no_padded_frames_end = 25 + 50 - (remainder if remainder != 0 else 50)

    start_frame = np.expand_dims(frames[0], 0)
    end_frame = np.expand_dims(frames[-1], 0)
    padded = np.concatenate(
        [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end,
        axis=0,
    )

    ptr = 0
    while ptr + 100 <= len(padded):
        yield padded[ptr : ptr + 100]
        ptr += 50


def predict_transnet(
    frames: np.ndarray,
    weights_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = TransNetV2()
    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    single_predictions = []
    many_predictions = []
    windows = list(iter_transnet_windows(frames))

    with torch.inference_mode():
        for index, window in enumerate(windows, start=1):
            inp = torch.from_numpy(window).unsqueeze(0).to(device)
            single_logits, extra = model(inp)
            single = torch.sigmoid(single_logits)[0, 25:75, 0].cpu().numpy()
            many = torch.sigmoid(extra["many_hot"])[0, 25:75, 0].cpu().numpy()
            single_predictions.append(single)
            many_predictions.append(many)
            print(f"Processed window {index}/{len(windows)}", end="\r", flush=True)

    print()
    single = np.concatenate(single_predictions)[: len(frames)]
    many = np.concatenate(many_predictions)[: len(frames)]
    return single, many


def predictions_to_scenes(predictions: np.ndarray, threshold: float) -> np.ndarray:
    binary = (predictions > threshold).astype(np.uint8)
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(binary):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t

    if t == 0:
        scenes.append([start, i])
    if not scenes:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


def merge_short_scenes(scenes: np.ndarray, min_frames: int) -> np.ndarray:
    if min_frames <= 1 or len(scenes) <= 1:
        return scenes

    merged = scenes.astype(np.int32).tolist()
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for idx, (start, end) in enumerate(merged):
            if end - start + 1 >= min_frames:
                continue

            if idx == 0:
                merged[1][0] = start
            elif idx == len(merged) - 1:
                merged[idx - 1][1] = end
            else:
                prev_len = merged[idx - 1][1] - merged[idx - 1][0] + 1
                next_len = merged[idx + 1][1] - merged[idx + 1][0] + 1
                if prev_len <= next_len:
                    merged[idx - 1][1] = end
                else:
                    merged[idx + 1][0] = start
            del merged[idx]
            changed = True
            break

    for idx in range(1, len(merged)):
        merged[idx][0] = merged[idx - 1][1] + 1
    return np.array(merged, dtype=np.int32)


def select_scores(single: np.ndarray, many: np.ndarray, score_mode: str) -> np.ndarray:
    if score_mode == "single":
        return single
    if score_mode == "many":
        return many
    if score_mode == "max":
        return np.maximum(single, many)
    if score_mode == "mean":
        return (single + many) / 2.0
    raise ValueError(f"Unknown score mode: {score_mode}")


def detect_transition_regions(
    selected: np.ndarray,
    single: np.ndarray,
    many: np.ndarray,
    low_threshold: float,
    peak_threshold: float,
    merge_gap: int,
) -> list[dict]:
    active = selected > low_threshold
    regions = []
    start = None
    for idx, value in enumerate(active):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            regions.append([start, idx - 1])
            start = None
    if start is not None:
        regions.append([start, len(selected) - 1])

    merged = []
    for start, end in regions:
        if not merged or start - merged[-1][1] > merge_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = end

    transitions = []
    for start, end in merged:
        window = selected[start : end + 1]
        peak_offset = int(np.argmax(window))
        peak_frame = start + peak_offset
        peak_score = float(selected[peak_frame])
        if peak_score < peak_threshold:
            continue

        single_peak = float(np.max(single[start : end + 1]))
        many_peak = float(np.max(many[start : end + 1]))
        length = end - start + 1
        if length <= 2 and single_peak >= many_peak:
            transition_type = "hard_cut"
        elif length >= 5 or many_peak > single_peak:
            transition_type = "gradual_or_complex"
        else:
            transition_type = "uncertain"

        transitions.append(
            {
                "transition_id": len(transitions) + 1,
                "start_frame": int(start),
                "end_frame": int(end),
                "peak_frame": int(peak_frame),
                "duration_frames": int(length),
                "peak_score": round(peak_score, 6),
                "single_peak": round(single_peak, 6),
                "many_hot_peak": round(many_peak, 6),
                "transition_type": transition_type,
            }
        )
    return transitions


def scenes_from_transitions(transitions: list[dict], frame_count: int) -> np.ndarray:
    boundaries = sorted(
        {
            int(t["peak_frame"])
            for t in transitions
            if 0 < int(t["peak_frame"]) < frame_count - 1
        }
    )
    scenes = []
    start = 0
    for boundary in boundaries:
        if boundary >= start:
            scenes.append([start, boundary])
            start = boundary + 1
    if start < frame_count:
        scenes.append([start, frame_count - 1])
    return np.array(scenes or [[0, frame_count - 1]], dtype=np.int32)


def write_transnet_outputs(
    output_dir: Path,
    video_path: Path,
    scenes: np.ndarray,
    predictions: np.ndarray,
    many_predictions: np.ndarray,
    fps: float,
    width: int,
    height: int,
    threshold: float,
    score_mode: str,
    min_shot_seconds: float,
) -> tuple[Path, Path, Path, list[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    csv_path = output_dir / f"{stem}_shots.csv"
    json_path = output_dir / f"{stem}_shots.json"
    pred_path = output_dir / f"{stem}_frame_predictions.csv"

    rows = []
    for shot_id, (start_frame, end_frame) in enumerate(scenes, start=1):
        start_seconds = start_frame / fps
        end_seconds = (end_frame + 1) / fps
        rows.append(
            {
                "shot_id": shot_id,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(end_seconds, 3),
                "duration_seconds": round(end_seconds - start_seconds, 3),
                "start_timecode": format_time(start_seconds),
                "end_timecode": format_time(end_seconds),
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "video": str(video_path),
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": len(predictions),
                "threshold": threshold,
                "score_mode": score_mode,
                "min_shot_seconds": min_shot_seconds,
                "shot_count": len(rows),
                "shots": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with pred_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "seconds", "single_frame_score", "many_hot_score"])
        for frame_idx, (single_score, many_score) in enumerate(zip(predictions, many_predictions)):
            writer.writerow([frame_idx, round(frame_idx / fps, 3), float(single_score), float(many_score)])

    return csv_path, json_path, pred_path, rows


def write_transitions(
    output_dir: Path,
    video_path: Path,
    transitions: list[dict],
    fps: float,
    low_threshold: float,
    peak_threshold: float,
) -> Path:
    path = output_dir / f"{video_path.stem}_transitions.csv"
    fields = [
        "transition_id",
        "start_frame",
        "end_frame",
        "peak_frame",
        "start_seconds",
        "end_seconds",
        "peak_seconds",
        "start_timecode",
        "end_timecode",
        "peak_timecode",
        "duration_frames",
        "duration_seconds",
        "peak_score",
        "single_peak",
        "many_hot_peak",
        "transition_type",
        "low_threshold",
        "peak_threshold",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in transitions:
            start_seconds = item["start_frame"] / fps
            end_seconds = (item["end_frame"] + 1) / fps
            peak_seconds = item["peak_frame"] / fps
            writer.writerow(
                {
                    **item,
                    "start_seconds": round(start_seconds, 3),
                    "end_seconds": round(end_seconds, 3),
                    "peak_seconds": round(peak_seconds, 3),
                    "start_timecode": format_time(start_seconds),
                    "end_timecode": format_time(end_seconds),
                    "peak_timecode": format_time(peak_seconds),
                    "duration_seconds": round(end_seconds - start_seconds, 3),
                    "low_threshold": low_threshold,
                    "peak_threshold": peak_threshold,
                }
            )
    return path


def read_source_frame(video_path: Path, frame_idx: int, target_width: int = 360) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = np.zeros((202, target_width, 3), dtype=np.uint8)
        cv2.putText(frame, f"Missing frame {frame_idx}", (16, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame

    height, width = frame.shape[:2]
    scale = target_width / max(1, width)
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def export_clips(video_path: Path, rows: list[dict], clips_dir: Path, ffmpeg_exe: str | None, clip_mode: str):
    if ffmpeg_exe is None:
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError("FFmpeg was not found; install ffmpeg or imageio_ffmpeg to export clips.") from exc

    clips_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        out = clips_dir / f"shot_{row['shot_id']:03d}_{row['start_timecode'].replace(':', '-')}.mp4"
        cmd = [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(row["start_seconds"]),
            "-to",
            str(row["end_seconds"]),
            "-i",
            str(video_path),
        ]
        if clip_mode == "copy":
            cmd += ["-c", "copy"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac"]
        cmd.append(str(out))
        subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# AutoShot stage helpers
# ---------------------------------------------------------------------------

def write_autoshot_predictions(path: Path, scores: list[float], fps: float) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "seconds", "single_frame_score", "many_hot_score"])
        for idx, score in enumerate(scores):
            writer.writerow([idx, round(idx / fps, 3), float(score), float(score)])


# ---------------------------------------------------------------------------
# Adaptive candidate sweep helpers
# ---------------------------------------------------------------------------

def read_video(video_path: Path, size=(160, 90)):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    smalls = []
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        smalls.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        frame_count += 1
    cap.release()
    if not frame_count:
        raise RuntimeError("No frames decoded.")
    return frame_count, np.asarray(smalls), float(fps)


def robust_norm(values, lo=50, hi=99):
    values = np.asarray(values, dtype=np.float32)
    a = float(np.percentile(values, lo))
    b = float(np.percentile(values, hi))
    if b <= a + 1e-8:
        return np.zeros_like(values)
    return np.clip((values - a) / (b - a), 0, 1)


def moving_max(values, radius=1):
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray([np.max(padded[i : i + 2 * radius + 1]) for i in range(len(values))], dtype=np.float32)


def load_model_scores(path: Path | None, frame_count: int):
    scores = np.zeros(frame_count, dtype=np.float32)
    if path is None or not path.exists():
        return scores
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["frame"])
            if idx >= frame_count:
                continue
            vals = []
            for key in ["single_frame_score", "many_hot_score"]:
                if key in row and row[key] != "":
                    vals.append(float(row[key]))
            if vals:
                scores[idx] = max(vals)
    return scores


def compute_signals(smalls):
    n = len(smalls)
    gray = np.asarray([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in smalls])
    edges = np.asarray([cv2.Canny(g, 80, 160) for g in gray])
    hsvs = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in smalls]
    hists = []
    for hsv in hsvs:
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 24], [0, 180, 0, 256])
        hists.append(cv2.normalize(hist, hist).flatten())
    hists = np.asarray(hists)

    hist_diff = np.zeros(n, dtype=np.float32)
    gray_diff = np.zeros(n, dtype=np.float32)
    edge_diff = np.zeros(n, dtype=np.float32)
    flash = np.zeros(n, dtype=np.float32)
    prepost = np.zeros(n, dtype=np.float32)

    brightness = np.asarray([np.mean(g) / 255.0 for g in gray], dtype=np.float32)
    saturation = np.asarray([np.mean(hsv[:, :, 1]) / 255.0 for hsv in hsvs], dtype=np.float32)
    flash[1:] = np.abs(np.diff(brightness))

    for i in range(1, n):
        hist_diff[i] = cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA)
        gray_diff[i] = np.mean(cv2.absdiff(gray[i], gray[i - 1])) / 255.0
        edge_diff[i] = np.mean(cv2.absdiff(edges[i], edges[i - 1])) / 255.0

    window = 4
    for i in range(window, n - window):
        pre_hist = np.mean(hists[i - window : i], axis=0)
        post_hist = np.mean(hists[i + 1 : i + 1 + window], axis=0)
        prepost[i] = cv2.compareHist(pre_hist.astype(np.float32), post_hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)

    return {
        "hist": robust_norm(hist_diff, 55, 99),
        "gray": robust_norm(gray_diff, 55, 99),
        "edge": robust_norm(edge_diff, 55, 99),
        "flash": robust_norm(flash, 70, 99.5),
        "prepost": robust_norm(prepost, 55, 99),
        "brightness": brightness,
        "saturation": saturation,
        "raw_hist": hist_diff,
        "raw_gray": gray_diff,
        "raw_edge": edge_diff,
        "raw_flash": flash,
        "raw_prepost": prepost,
    }


def local_peaks(score, threshold, radius):
    peaks = []
    for i in range(radius, len(score) - radius):
        if score[i] < threshold:
            continue
        if score[i] >= np.max(score[i - radius : i + radius + 1]):
            peaks.append(i)
    return peaks


def suppress(peaks, score, min_gap):
    ordered = sorted(peaks, key=lambda i: float(score[i]), reverse=True)
    selected = []
    for peak in ordered:
        if all(abs(peak - kept) > min_gap for kept in selected):
            selected.append(peak)
    return sorted(selected)


def build_candidates(
    transnet,
    autoshot,
    signals,
    fps,
    threshold,
    min_gap_frames,
    model_keep_threshold=0.30,
    visual_keep_threshold=0.58,
    weak_visual_keep_threshold=0.46,
    min_supports=2,
):
    visual = np.clip(
        0.30 * signals["hist"]
        + 0.20 * signals["gray"]
        + 0.18 * signals["edge"]
        + 0.17 * signals["flash"]
        + 0.15 * signals["prepost"],
        0,
        1,
    )
    model = np.maximum(transnet, autoshot)
    score = np.maximum.reduce([moving_max(model, 1), moving_max(visual, 1)])
    peaks = local_peaks(score, threshold, radius=max(1, int(round(fps * 0.08))))
    peaks = suppress(peaks, score, min_gap_frames)
    candidates = []
    for idx, peak in enumerate(peaks, start=1):
        local = slice(max(0, peak - 2), min(len(score), peak + 3))
        model_peak = float(np.max(model[local]))
        transnet_peak = float(np.max(transnet[local]))
        autoshot_peak = float(np.max(autoshot[local]))
        signal_peaks = {name: float(np.max(signals[name][local])) for name in ["hist", "gray", "edge", "flash", "prepost"]}
        visual_peak = max(signal_peaks.values())
        visual_supports = sum(1 for value in signal_peaks.values() if value >= weak_visual_keep_threshold)
        has_structural_change = signal_peaks["prepost"] >= weak_visual_keep_threshold
        has_instant_change = max(signal_peaks["hist"], signal_peaks["gray"], signal_peaks["edge"], signal_peaks["flash"]) >= weak_visual_keep_threshold

        keep = False
        if model_peak >= model_keep_threshold:
            keep = True
        elif visual_peak >= visual_keep_threshold and visual_supports >= min_supports and has_structural_change and has_instant_change:
            keep = True
        if not keep:
            continue

        reasons = []
        if transnet_peak >= model_keep_threshold:
            reasons.append("transnet")
        if autoshot_peak >= model_keep_threshold:
            reasons.append("autoshot")
        for name, value in signal_peaks.items():
            if value >= weak_visual_keep_threshold:
                reasons.append(name)
        if not reasons:
            best_signal = max(["hist", "gray", "edge", "flash", "prepost"], key=lambda name: signals[name][peak])
            reasons.append(best_signal)

        if signal_peaks["flash"] >= weak_visual_keep_threshold and (
            signal_peaks["gray"] >= weak_visual_keep_threshold or signal_peaks["hist"] >= weak_visual_keep_threshold
        ):
            transition_kind = "flash_or_luma"
        elif signal_peaks["prepost"] >= weak_visual_keep_threshold and visual_supports >= min_supports:
            transition_kind = "complex_visual"
        else:
            transition_kind = "hard_cut"
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "peak_frame": int(peak),
                "peak_timecode": format_time(peak / fps),
                "peak_seconds": round(peak / fps, 3),
                "score": round(float(score[peak]), 6),
                "transnet_score": round(transnet_peak, 6),
                "autoshot_score": round(autoshot_peak, 6),
                "hist_score": round(signal_peaks["hist"], 6),
                "gray_score": round(signal_peaks["gray"], 6),
                "edge_score": round(signal_peaks["edge"], 6),
                "flash_score": round(signal_peaks["flash"], 6),
                "prepost_score": round(signal_peaks["prepost"], 6),
                "visual_supports": visual_supports,
                "transition_kind": transition_kind,
                "reason": "+".join(reasons),
            }
        )
    return candidates, score, visual


def ffmpeg_path():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def export_candidate_clips(video_path, out_dir, candidates, fps, clip_seconds):
    exe = ffmpeg_path()
    clips_dir = out_dir / "candidate_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old in clips_dir.glob("candidate_*.mp4"):
        old.unlink()
    half = clip_seconds / 2.0
    rows = []
    for c in candidates:
        peak_seconds = c["peak_frame"] / fps
        start = max(0.0, peak_seconds - half)
        end = peak_seconds + half
        path = clips_dir / f"candidate_{c['candidate_id']:03d}_{c['peak_timecode'].replace(':', '-')}.mp4"
        cmd = [
            exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.6f}",
            "-to",
            f"{end:.6f}",
            "-i",
            str(video_path),
            "-vf",
            "scale=480:-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(path),
        ]
        subprocess.run(cmd, check=True)
        rows.append({**c, "clip": str(path), "clip_start": round(start, 3), "clip_end": round(end, 3)})
    return rows


def find_luma_intervals(brightness, dark_threshold, bright_threshold, pad_frames):
    active = (brightness <= dark_threshold) | (brightness >= bright_threshold)
    intervals = []
    start = None
    for idx, value in enumerate(active):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            intervals.append([max(0, start - pad_frames), min(len(active) - 1, idx - 1 + pad_frames)])
            start = None
    if start is not None:
        intervals.append([max(0, start - pad_frames), len(active) - 1])
    return intervals


def expand_candidate_interval(
    candidate,
    frame_count,
    fps,
    signals,
    hard_pad_frames,
    complex_threshold,
    flash_threshold,
    max_complex_seconds,
):
    peak = int(candidate["peak_frame"])
    kind = candidate.get("transition_kind", "hard_cut")

    if kind == "hard_cut":
        start = max(0, peak - hard_pad_frames)
        end = min(frame_count - 1, peak + hard_pad_frames)
        return start, end, "hard_cut_pad"

    if kind == "flash_or_luma":
        active = np.maximum.reduce([signals["flash"], signals["gray"], signals["hist"]]) >= flash_threshold
        source = "flash_luma_expand"
    else:
        active = np.maximum.reduce([signals["hist"], signals["gray"], signals["edge"], signals["prepost"]]) >= complex_threshold
        source = "complex_visual_expand"

    max_frames = max(hard_pad_frames, int(round(max_complex_seconds * fps)))
    start = peak
    while start > 0 and peak - start < max_frames and active[start - 1]:
        start -= 1
    end = peak
    while end < frame_count - 1 and end - peak < max_frames and active[end + 1]:
        end += 1

    start = max(0, start - 1)
    end = min(frame_count - 1, end + 1)
    if end - start + 1 <= hard_pad_frames * 2 + 1:
        start = max(0, peak - hard_pad_frames)
        end = min(frame_count - 1, peak + hard_pad_frames)
    return start, end, source


def build_runs_from_candidates(
    candidates,
    frame_count,
    fps,
    signals=None,
    transition_seconds=0.45,
    hard_pad_frames=2,
    complex_threshold=0.45,
    flash_threshold=0.45,
    max_complex_seconds=0.75,
    brightness=None,
    dark_threshold=0.08,
    bright_threshold=0.96,
    luma_pad_frames=2,
    min_normal_seconds=0.45,
):
    intervals = []
    for candidate in candidates:
        if signals is None:
            half = max(1, int(round((transition_seconds * fps) / 2.0)))
            start = max(0, int(candidate["peak_frame"]) - half)
            end = min(frame_count - 1, int(candidate["peak_frame"]) + half)
            source = "fixed_window"
        else:
            start, end, source = expand_candidate_interval(
                candidate,
                frame_count,
                fps,
                signals,
                hard_pad_frames=hard_pad_frames,
                complex_threshold=complex_threshold,
                flash_threshold=flash_threshold,
                max_complex_seconds=max_complex_seconds,
            )
        intervals.append([start, end, source])

    if brightness is not None:
        intervals.extend([start, end, "black_white_luma"] for start, end in find_luma_intervals(brightness, dark_threshold, bright_threshold, luma_pad_frames))

    intervals.sort(key=lambda item: item[0])
    merged_intervals = []
    for start, end, source in intervals:
        if not merged_intervals or start > merged_intervals[-1][1] + 1:
            merged_intervals.append([start, end, {source}])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], end)
            merged_intervals[-1][2].add(source)

    min_normal_frames = max(1, int(round(min_normal_seconds * fps)))
    compacted = []
    for interval in merged_intervals:
        if not compacted:
            compacted.append(interval)
            continue
        gap = interval[0] - compacted[-1][1] - 1
        if gap < min_normal_frames:
            compacted[-1][1] = max(compacted[-1][1], interval[1])
            compacted[-1][2].update(interval[2])
            compacted[-1][2].add("short_normal_gap_absorbed")
        else:
            compacted.append(interval)
    merged_intervals = compacted

    runs = []
    cursor = 0
    for start, end, sources in merged_intervals:
        if cursor < start:
            runs.append({"label": "normal", "start_frame": cursor, "end_frame": start - 1, "source": ""})
        runs.append({"label": "transition", "start_frame": start, "end_frame": end, "source": "+".join(sorted(sources))})
        cursor = end + 1
    if cursor < frame_count:
        runs.append({"label": "normal", "start_frame": cursor, "end_frame": frame_count - 1, "source": ""})
    return runs


def export_run_clips(video_path, out_dir, runs, fps):
    exe = ffmpeg_path()
    clips_dir = out_dir / "run_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old in clips_dir.glob("run_*.mp4"):
        old.unlink()

    rows = []
    for idx, run in enumerate(runs, start=1):
        start = run["start_frame"] / fps
        end = (run["end_frame"] + 1) / fps
        out = clips_dir / f"run_{idx:03d}_{run['label']}_{format_time(start).replace(':', '-')}.mp4"
        cmd = [
            exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.6f}",
            "-to",
            f"{end:.6f}",
            "-i",
            str(video_path),
            "-vf",
            "scale=480:-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(out),
        ]
        subprocess.run(cmd, check=True)
        rows.append(
            {
                "run_id": idx,
                "label": run["label"],
                "start_frame": run["start_frame"],
                "end_frame": run["end_frame"],
                "start_timecode": format_time(start),
                "end_timecode": format_time(end),
                "duration_seconds": round(end - start, 3),
                "source": run.get("source", ""),
                "clip": str(out),
            }
        )
    return rows


def output_path(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target, base_dir).replace(os.sep, "/")


def fourcc_to_text(value: float) -> str | None:
    code = int(value)
    if code <= 0:
        return None
    chars = [chr((code >> 8 * i) & 0xFF) for i in range(4)]
    text = "".join(chars).strip()
    return text or None


def probe_video_metadata(video_path: Path, fps: float, frame_count: int) -> dict:
    metadata = {
        "path": str(video_path),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "duration_seconds": round(frame_count / fps, 3) if fps else None,
        "width": None,
        "height": None,
        "codec_name": None,
        "has_audio": None,
    }

    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        metadata["codec_name"] = fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return metadata

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,width,height",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
        streams = json.loads(result.stdout).get("streams", [])
    except Exception:
        return metadata

    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video_stream:
        metadata["codec_name"] = video_stream.get("codec_name") or metadata["codec_name"]
        metadata["width"] = video_stream.get("width") or metadata["width"]
        metadata["height"] = video_stream.get("height") or metadata["height"]
    metadata["has_audio"] = any(stream.get("codec_type") == "audio" for stream in streams)
    return metadata


def write_runs_csv(out_dir, run_rows):
    csv_path = out_dir / "normal_transition_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["id", "type", "start_frame", "end_frame", "start_timecode", "end_timecode", "duration_sec", "source", "clip_path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in run_rows:
            writer.writerow(
                {
                    "id": row["run_id"],
                    "type": row["label"],
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "start_timecode": row["start_timecode"],
                    "end_timecode": row["end_timecode"],
                    "duration_sec": row["duration_seconds"],
                    "source": row.get("source", ""),
                    "clip_path": row["clip"],
                }
            )
    return csv_path


def build_summary(out_dir, video_path, report_path, run_rows, fps, frame_count, threshold, runs_csv_path=None):
    summary_path = out_dir / "summary.json"
    base_dir = summary_path.parent
    video_metadata = probe_video_metadata(video_path, fps, frame_count)
    video_metadata["filename"] = video_path.name
    video_metadata["path"] = video_path.name

    def clip_item(row):
        clip_path = Path(row["clip"])
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        start_sec = start_frame / fps if fps else None
        end_sec = (end_frame + 1) / fps if fps else None
        return {
            "id": int(row["run_id"]),
            "type": row["label"],
            "start_sec": round(start_sec, 3) if start_sec is not None else None,
            "end_sec": round(end_sec, 3) if end_sec is not None else None,
            "duration_sec": float(row["duration_seconds"]),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_timecode": row["start_timecode"],
            "end_timecode": row["end_timecode"],
            "source": row.get("source", ""),
            "clip_path": output_path(base_dir, clip_path),
        }

    clips = [clip_item(row) for row in run_rows]
    transition_clips = [clip_item(row) for row in run_rows if row["label"] == "transition"]
    normal_clips = [clip_item(row) for row in run_rows if row["label"] == "normal"]
    outputs = {
        "report_html": output_path(base_dir, report_path),
        "run_clips_dir": output_path(base_dir, out_dir / "run_clips"),
    }
    if runs_csv_path is not None:
        outputs["runs_csv"] = output_path(base_dir, runs_csv_path)

    return {
        "schema_version": "1.0",
        "video": video_metadata,
        "parameters": {
            "candidate_threshold": float(threshold),
        },
        "outputs": outputs,
        "counts": {
            "total_clips": len(run_rows),
            "transition_clips": len(transition_clips),
            "normal_clips": len(normal_clips),
        },
        "clips": clips,
        "clip_groups": {
            "transition": transition_clips,
            "normal": normal_clips,
        },
    }


def write_summary_json(out_dir, summary):
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def cleanup_debug_outputs(out_dir: Path) -> None:
    for file_name in ["candidate_sweep_report.html", "candidate_sweep.csv", "candidate_sweep_meta.json"]:
        path = out_dir / file_name
        if path.exists():
            path.unlink()
    clips_dir = out_dir / "candidate_clips"
    if clips_dir.exists():
        shutil.rmtree(clips_dir)


def write_synthetic_video(video_path: Path, fps: float = 10.0) -> int:
    width, height = 160, 90
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create smoke test video: {video_path}")

    colors = [(30, 90, 180), (240, 240, 240), (180, 60, 40)]
    frame_count = 0
    for color in colors:
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        for _ in range(10):
            writer.write(frame)
            frame_count += 1
    writer.release()
    return frame_count


def run_pipeline(video, output_dir="outputs_pipeline", device="cpu", debug=False):
    """库入口：对单个视频跑完整三步管线，返回 result 目录下 summary.json 的路径。"""
    from video_shot_splitter.cli import run_pipeline_cli
    argv = [str(video), "--output-dir", str(output_dir), "--device", device]
    if debug:
        argv.append("--debug")
    run_pipeline_cli(argv)
    from pathlib import Path
    return Path(output_dir) / "result" / "summary.json"
    return frame_count
