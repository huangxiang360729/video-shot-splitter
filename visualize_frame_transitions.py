import argparse
import csv
import html
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np


def format_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def read_video(video_path: Path, thumb_width: int = 240):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    thumbs = []
    smalls = []

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
        scale = thumb_width / max(1, bgr.shape[1])
        thumb_height = max(1, int(round(bgr.shape[0] * scale)))
        thumbs.append(cv2.resize(bgr, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA))
        smalls.append(cv2.resize(bgr, (160, 90), interpolation=cv2.INTER_AREA))

    cap.release()
    if not frames:
        raise RuntimeError("No frames decoded.")
    return frames, thumbs, np.asarray(smalls), float(fps), source_width, source_height


def load_predictions(path: Path, frame_count: int):
    single = np.zeros(frame_count, dtype=np.float32)
    many = np.zeros(frame_count, dtype=np.float32)
    if path is None or not path.exists():
        return single, many

    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            idx = int(row["frame"])
            if idx < frame_count:
                single[idx] = float(row["single_frame_score"])
                many[idx] = float(row["many_hot_score"])
    return single, many


def robust_normalize(values: np.ndarray, lo_pct: float = 50, hi_pct: float = 99) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(np.percentile(values, lo_pct))
    hi = float(np.percentile(values, hi_pct))
    if hi <= lo + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0, 1)


def compute_visual_change(smalls: np.ndarray):
    frame_count = len(smalls)
    gray = np.asarray([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in smalls])
    edges = np.asarray([cv2.Canny(g, 80, 160) for g in gray])

    histograms = []
    for frame in smalls:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    histograms = np.asarray(histograms)

    gray_diff = np.zeros(frame_count, dtype=np.float32)
    edge_diff = np.zeros(frame_count, dtype=np.float32)
    hist_diff = np.zeros(frame_count, dtype=np.float32)

    for idx in range(1, frame_count):
        gray_diff[idx] = np.mean(cv2.absdiff(gray[idx], gray[idx - 1])) / 255.0
        edge_diff[idx] = np.mean(cv2.absdiff(edges[idx], edges[idx - 1])) / 255.0
        hist_diff[idx] = cv2.compareHist(histograms[idx - 1], histograms[idx], cv2.HISTCMP_BHATTACHARYYA)

    visual_score = (
        0.50 * robust_normalize(hist_diff, 55, 99)
        + 0.30 * robust_normalize(gray_diff, 55, 99)
        + 0.20 * robust_normalize(edge_diff, 55, 99)
    )
    return {
        "visual_score": np.clip(visual_score, 0, 1),
        "hist_diff": hist_diff,
        "gray_diff": gray_diff,
        "edge_diff": edge_diff,
    }


def smooth_scores(scores: np.ndarray):
    padded = np.pad(scores, (1, 1), mode="edge")
    return np.asarray([np.max(padded[i : i + 3]) for i in range(len(scores))], dtype=np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def frame_descriptor(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_hs = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256]).flatten()
    hist_hs = hist_hs / (np.linalg.norm(hist_hs) + 1e-8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (24, 14), interpolation=cv2.INTER_AREA).astype(np.float32).flatten() / 255.0
    gray_small = gray_small - float(np.mean(gray_small))
    gray_small = gray_small / (np.linalg.norm(gray_small) + 1e-8)

    edge = cv2.Canny(gray, 80, 160)
    edge_small = cv2.resize(edge, (24, 14), interpolation=cv2.INTER_AREA).astype(np.float32).flatten() / 255.0
    edge_small = edge_small / (np.linalg.norm(edge_small) + 1e-8)
    return np.concatenate([hist_hs * 0.60, gray_small * 0.25, edge_small * 0.15]).astype(np.float32)


def average_descriptor(smalls: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, start)
    end = min(len(smalls) - 1, end)
    if end < start:
        return frame_descriptor(smalls[max(0, min(len(smalls) - 1, start))])
    descs = [frame_descriptor(smalls[idx]) for idx in range(start, end + 1)]
    return np.mean(descs, axis=0)


def motion_compensated_residual(prev_frame: np.ndarray, next_frame: np.ndarray) -> tuple[float, float]:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=160, qualityLevel=0.01, minDistance=6)
    if pts is None or len(pts) < 12:
        return 1.0, 0.0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, pts, None)
    if next_pts is None or status is None:
        return 1.0, 0.0

    good_prev = pts[status.flatten() == 1].reshape(-1, 2)
    good_next = next_pts[status.flatten() == 1].reshape(-1, 2)
    if len(good_prev) < 12:
        return 1.0, 0.0

    matrix, inliers = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=3)
    if matrix is None or inliers is None:
        return 1.0, 0.0

    warped = cv2.warpAffine(prev_gray, matrix, (prev_gray.shape[1], prev_gray.shape[0]))
    residual = float(np.mean(cv2.absdiff(warped, next_gray)) / 255.0)
    inlier_ratio = float(np.mean(inliers))
    return min(1.0, residual * 4.0), inlier_ratio


def local_maxima(scores: np.ndarray, threshold: float) -> list[int]:
    peaks = []
    for idx in range(1, len(scores) - 1):
        if scores[idx] >= threshold and scores[idx] >= scores[idx - 1] and scores[idx] >= scores[idx + 1]:
            peaks.append(idx)
    return peaks


def non_max_suppress(peaks: list[int], scores: np.ndarray, radius: int) -> list[int]:
    ordered = sorted(peaks, key=lambda idx: float(scores[idx]), reverse=True)
    selected = []
    for peak in ordered:
        if all(abs(peak - kept) > radius for kept in selected):
            selected.append(peak)
    return sorted(selected)


def label_transition_frames_rerank(
    smalls: np.ndarray,
    model_score: np.ndarray,
    many_score: np.ndarray,
    visual_score: np.ndarray,
    candidate_threshold: float,
    accept_threshold: float,
    prepost_threshold: float,
    window: int,
    guard: int,
    nms_radius: int,
    hard_pad_frames: int,
    gradual_expand_threshold: float,
    max_expand_frames: int,
):
    fused = np.clip(0.78 * model_score + 0.22 * visual_score, 0, 1)
    model_smooth = smooth_scores(model_score)
    many_smooth = smooth_scores(many_score)
    visual_smooth = smooth_scores(visual_score)
    candidate_score = np.maximum.reduce([model_smooth, many_smooth * 0.95, visual_smooth * 0.55])
    candidates = local_maxima(candidate_score, candidate_threshold)
    candidates = non_max_suppress(candidates, candidate_score, nms_radius)

    labels = np.zeros(len(fused), dtype=np.uint8)
    transitions = []
    used_ranges = []
    for peak in candidates:
        pre_desc = average_descriptor(smalls, peak - window, peak - guard)
        post_desc = average_descriptor(smalls, peak + guard, peak + window)
        prepost_distance = cosine_distance(pre_desc, post_desc)
        prepost_distance = float(np.clip(prepost_distance * 1.75, 0, 1))

        left = max(0, peak - 2)
        right = min(len(smalls) - 1, peak + 2)
        motion_residual, motion_inlier_ratio = motion_compensated_residual(smalls[left], smalls[right])

        region_start = max(0, peak - max(guard, 3))
        region_end = min(len(fused) - 1, peak + max(guard, 3))
        region_model_peak = float(np.max(model_smooth[region_start : region_end + 1]))
        region_many_peak = float(np.max(many_smooth[max(0, peak - 8) : min(len(fused), peak + 9)]))
        region_visual_peak = float(np.max(visual_smooth[region_start : region_end + 1]))

        rerank_score = (
            0.48 * region_model_peak
            + 0.18 * region_many_peak
            + 0.24 * prepost_distance
            + 0.10 * motion_residual
        )
        camera_motion_like = motion_inlier_ratio >= 0.46 and motion_residual <= 0.30 and region_many_peak < 0.62

        if rerank_score < accept_threshold:
            continue
        if prepost_distance < prepost_threshold and region_many_peak < 0.65:
            continue
        if camera_motion_like and region_model_peak < 0.72:
            continue

        is_gradual = region_many_peak >= max(0.62, region_model_peak * 0.92)
        if is_gradual:
            start = peak
            while start > 0 and peak - start < max_expand_frames:
                if many_smooth[start - 1] < gradual_expand_threshold and fused[start - 1] < gradual_expand_threshold:
                    break
                start -= 1
            end = peak
            while end < len(fused) - 1 and end - peak < max_expand_frames:
                if many_smooth[end + 1] < gradual_expand_threshold and fused[end + 1] < gradual_expand_threshold:
                    break
                end += 1
        else:
            start = max(0, peak - hard_pad_frames)
            end = min(len(fused) - 1, peak + hard_pad_frames)

        if any(not (end < s - nms_radius or start > e + nms_radius) for s, e in used_ranges):
            continue
        used_ranges.append((start, end))
        labels[start : end + 1] = 1
        transitions.append(
            {
                "transition_id": len(transitions) + 1,
                "start_frame": int(start),
                "end_frame": int(end),
                "peak_frame": int(peak),
                "duration_frames": int(end - start + 1),
                "peak_fused_score": round(float(rerank_score), 6),
                "peak_model_score": round(float(model_score[peak]), 6),
                "peak_many_hot_score": round(float(many_score[peak]), 6),
                "peak_visual_score": round(float(visual_score[peak]), 6),
                "region_model_peak": round(region_model_peak, 6),
                "region_many_hot_peak": round(region_many_peak, 6),
                "region_visual_peak": round(region_visual_peak, 6),
                "prepost_distance": round(prepost_distance, 6),
                "motion_residual": round(motion_residual, 6),
                "motion_inlier_ratio": round(motion_inlier_ratio, 6),
                "transition_type": "gradual_or_complex" if is_gradual else "hard_cut",
            }
        )

    transitions.sort(key=lambda item: item["start_frame"])
    for idx, item in enumerate(transitions, start=1):
        item["transition_id"] = idx
    return labels, transitions, fused


def label_transition_frames(
    model_score: np.ndarray,
    many_score: np.ndarray,
    visual_score: np.ndarray,
    low_threshold: float,
    peak_threshold: float,
    merge_gap: int,
    pad_frames: int,
):
    fused = np.clip(0.62 * model_score + 0.38 * visual_score, 0, 1)
    fused = smooth_scores(fused)
    active = (fused >= low_threshold) | (many_score >= low_threshold + 0.08)

    intervals = []
    start = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            intervals.append([start, idx - 1])
            start = None
    if start is not None:
        intervals.append([start, len(active) - 1])

    merged = []
    for start, end in intervals:
        if not merged or start - merged[-1][1] > merge_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = end

    labels = np.zeros(len(active), dtype=np.uint8)
    transitions = []
    for start, end in merged:
        peak = start + int(np.argmax(fused[start : end + 1]))
        peak_score = float(fused[peak])
        if peak_score < peak_threshold:
            continue
        padded_start = max(0, start - pad_frames)
        padded_end = min(len(active) - 1, end + pad_frames)
        labels[padded_start : padded_end + 1] = 1

        region_model_peak = float(np.max(model_score[padded_start : padded_end + 1]))
        region_many_peak = float(np.max(many_score[padded_start : padded_end + 1]))
        region_visual_peak = float(np.max(visual_score[padded_start : padded_end + 1]))
        length = padded_end - padded_start + 1
        if length <= 8 and region_visual_peak >= 0.45 and region_many_peak < 0.65:
            kind = "hard_cut"
        elif length >= 5 or region_many_peak >= region_model_peak:
            kind = "gradual_or_complex"
        else:
            kind = "uncertain"

        transitions.append(
            {
                "transition_id": len(transitions) + 1,
                "start_frame": int(padded_start),
                "end_frame": int(padded_end),
                "peak_frame": int(peak),
                "duration_frames": int(length),
                "peak_fused_score": round(peak_score, 6),
                "peak_model_score": round(float(model_score[peak]), 6),
                "peak_many_hot_score": round(float(many_score[peak]), 6),
                "peak_visual_score": round(float(visual_score[peak]), 6),
                "region_model_peak": round(region_model_peak, 6),
                "region_many_hot_peak": round(region_many_peak, 6),
                "region_visual_peak": round(region_visual_peak, 6),
                "transition_type": kind,
            }
        )

    return labels, transitions, fused


def label_transition_frames_anchored(
    model_score: np.ndarray,
    many_score: np.ndarray,
    visual_score: np.ndarray,
    model_peak_threshold: float,
    weak_model_threshold: float,
    visual_peak_threshold: float,
    expansion_threshold: float,
    merge_gap: int,
    hard_pad_frames: int,
    max_expand_frames: int,
):
    fused = np.clip(0.72 * model_score + 0.28 * visual_score, 0, 1)
    smoothed_model = smooth_scores(model_score)
    smoothed_many = smooth_scores(many_score)
    smoothed_visual = smooth_scores(visual_score)

    candidates = []
    for idx in range(1, len(fused) - 1):
        is_model_peak = smoothed_model[idx] >= smoothed_model[idx - 1] and smoothed_model[idx] >= smoothed_model[idx + 1]
        is_visual_peak = smoothed_visual[idx] >= smoothed_visual[idx - 1] and smoothed_visual[idx] >= smoothed_visual[idx + 1]
        strong_model = is_model_peak and smoothed_model[idx] >= model_peak_threshold
        model_supported_visual = (
            is_visual_peak
            and smoothed_visual[idx] >= visual_peak_threshold
            and np.max(smoothed_model[max(0, idx - 4) : min(len(fused), idx + 5)]) >= weak_model_threshold
        )
        if strong_model or model_supported_visual:
            candidates.append(idx)

    candidates = sorted(candidates, key=lambda i: float(max(smoothed_model[i], smoothed_visual[i])), reverse=True)
    selected = []
    for idx in candidates:
        if all(abs(idx - prev) > merge_gap for prev in selected):
            selected.append(idx)
    selected.sort()

    labels = np.zeros(len(fused), dtype=np.uint8)
    transitions = []
    used_ranges = []
    for peak in selected:
        model_peak = float(np.max(smoothed_model[max(0, peak - 2) : min(len(fused), peak + 3)]))
        many_peak = float(np.max(smoothed_many[max(0, peak - 4) : min(len(fused), peak + 5)]))
        visual_peak = float(np.max(smoothed_visual[max(0, peak - 2) : min(len(fused), peak + 3)]))

        is_gradual = many_peak >= model_peak or many_peak >= 0.62
        if is_gradual:
            start = peak
            while start > 0 and peak - start < max_expand_frames:
                if fused[start - 1] < expansion_threshold and smoothed_many[start - 1] < expansion_threshold:
                    break
                start -= 1
            end = peak
            while end < len(fused) - 1 and end - peak < max_expand_frames:
                if fused[end + 1] < expansion_threshold and smoothed_many[end + 1] < expansion_threshold:
                    break
                end += 1
        else:
            start = max(0, peak - hard_pad_frames)
            end = min(len(fused) - 1, peak + hard_pad_frames)

        if any(not (end < s - merge_gap or start > e + merge_gap) for s, e in used_ranges):
            continue
        used_ranges.append((start, end))
        labels[start : end + 1] = 1

        transition_type = "gradual_or_complex" if is_gradual else "hard_cut"
        transitions.append(
            {
                "transition_id": len(transitions) + 1,
                "start_frame": int(start),
                "end_frame": int(end),
                "peak_frame": int(peak),
                "duration_frames": int(end - start + 1),
                "peak_fused_score": round(float(fused[peak]), 6),
                "peak_model_score": round(float(model_score[peak]), 6),
                "peak_many_hot_score": round(float(many_score[peak]), 6),
                "peak_visual_score": round(float(visual_score[peak]), 6),
                "region_model_peak": round(model_peak, 6),
                "region_many_hot_peak": round(many_peak, 6),
                "region_visual_peak": round(visual_peak, 6),
                "transition_type": transition_type,
            }
        )

    transitions.sort(key=lambda item: item["start_frame"])
    for idx, item in enumerate(transitions, start=1):
        item["transition_id"] = idx
    return labels, transitions, fused


def build_runs(labels: np.ndarray):
    runs = []
    start = 0
    current = int(labels[0])
    for idx in range(1, len(labels)):
        if int(labels[idx]) != current:
            runs.append({"label": "transition" if current else "normal", "start_frame": start, "end_frame": idx - 1})
            start = idx
            current = int(labels[idx])
    runs.append({"label": "transition" if current else "normal", "start_frame": start, "end_frame": len(labels) - 1})
    return runs


def draw_card(image: np.ndarray, title: str, subtitle: str, label: str):
    color = (54, 166, 118) if label == "normal" else (196, 62, 58)
    label_bg = np.full((28, image.shape[1], 3), color, dtype=np.uint8)
    cv2.putText(label_bg, label.upper(), (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    footer = np.full((52, image.shape[1], 3), 248, dtype=np.uint8)
    cv2.putText(footer, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (28, 32, 36), 1, cv2.LINE_AA)
    cv2.putText(footer, subtitle, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (92, 98, 105), 1, cv2.LINE_AA)
    return np.vstack([label_bg, image, footer])


def save_storyboard(
    out_path: Path,
    thumbs: list[np.ndarray],
    runs: list[dict],
    fps: float,
    fused: np.ndarray,
    max_transition_frames: int,
):
    cards = []
    for run in runs:
        start, end = run["start_frame"], run["end_frame"]
        if run["label"] == "normal":
            frame_ids = [(start + end) // 2]
        else:
            frame_ids = list(range(start, end + 1))
            if len(frame_ids) > max_transition_frames:
                picks = np.linspace(start, end, max_transition_frames).round().astype(int)
                frame_ids = sorted(set(int(x) for x in picks))

        for pos, frame_id in enumerate(frame_ids):
            if run["label"] == "normal":
                title = f"normal run f{start}-{end}"
                subtitle = f"representative f{frame_id}  {format_time(frame_id / fps)}  score {fused[frame_id]:.3f}"
            else:
                title = f"transition f{start}-{end}"
                subtitle = f"frame f{frame_id}  {format_time(frame_id / fps)}  score {fused[frame_id]:.3f}"
                if pos == len(frame_ids) - 1 and end - start + 1 > len(frame_ids):
                    subtitle += "  sampled"
            cards.append(draw_card(thumbs[frame_id], title, subtitle, run["label"]))

    cols = 4
    gap = 12
    card_w = cards[0].shape[1]
    card_h = cards[0].shape[0]
    rows = int(np.ceil(len(cards) / cols))
    canvas = np.full((rows * card_h + (rows + 1) * gap, cols * card_w + (cols + 1) * gap, 3), 238, dtype=np.uint8)
    for idx, card in enumerate(cards):
        y = gap + (idx // cols) * (card_h + gap)
        x = gap + (idx % cols) * (card_w + gap)
        canvas[y : y + card_h, x : x + card_w] = card
    cv2.imwrite(str(out_path), canvas)


def relative_url(from_file: Path, target: Path) -> str:
    rel = os.path.relpath(target, from_file.parent).replace(os.sep, "/")
    return quote(rel, safe="/:._-")


def export_run_clips(video_path: Path, output_dir: Path, stem: str, runs: list[dict], fps: float):
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("FFmpeg is required to export run clips.") from exc

    clips_dir = output_dir / f"{stem}_run_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for stale_clip in clips_dir.glob("run_*.mp4"):
        stale_clip.unlink()
    clip_rows = []
    for idx, run in enumerate(runs, start=1):
        start_seconds = run["start_frame"] / fps
        end_seconds = (run["end_frame"] + 1) / fps
        start_tc = format_time(start_seconds).replace(":", "-")
        out = clips_dir / f"run_{idx:03d}_{run['label']}_{start_tc}.mp4"
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_seconds:.6f}",
            "-to",
            f"{end_seconds:.6f}",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            str(out),
        ]
        subprocess.run(cmd, check=True)
        clip_rows.append(
            {
                "run_id": idx,
                "label": run["label"],
                "start_frame": run["start_frame"],
                "end_frame": run["end_frame"],
                "start_timecode": format_time(start_seconds),
                "end_timecode": format_time(end_seconds),
                "clip": str(out),
            }
        )
    return clips_dir, clip_rows


def write_csvs(output_dir: Path, stem: str, labels, transitions, runs, fps, model, many, visual, fused):
    labels_path = output_dir / f"{stem}_frame_labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timecode", "seconds", "label", "fused_score", "model_score", "many_hot_score", "visual_score"])
        for idx, label in enumerate(labels):
            writer.writerow(
                [
                    idx,
                    format_time(idx / fps),
                    round(idx / fps, 3),
                    "transition" if label else "normal",
                    float(fused[idx]),
                    float(model[idx]),
                    float(many[idx]),
                    float(visual[idx]),
                ]
            )

    transitions_path = output_dir / f"{stem}_transition_frames.csv"
    with transitions_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "transition_id",
            "start_frame",
            "end_frame",
            "peak_frame",
            "start_timecode",
            "end_timecode",
            "peak_timecode",
            "duration_frames",
            "peak_fused_score",
            "peak_model_score",
            "peak_many_hot_score",
            "peak_visual_score",
            "region_model_peak",
            "region_many_hot_peak",
            "region_visual_peak",
            "prepost_distance",
            "motion_residual",
            "motion_inlier_ratio",
            "transition_type",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in transitions:
            row = {field: item.get(field, "") for field in fields}
            row.update(
                {
                    "start_timecode": format_time(item["start_frame"] / fps),
                    "end_timecode": format_time((item["end_frame"] + 1) / fps),
                    "peak_timecode": format_time(item["peak_frame"] / fps),
                }
            )
            writer.writerow(
                row
            )

    runs_path = output_dir / f"{stem}_collapsed_runs.csv"
    with runs_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "label", "start_frame", "end_frame", "start_timecode", "end_timecode", "duration_seconds"])
        for idx, run in enumerate(runs, start=1):
            writer.writerow(
                [
                    idx,
                    run["label"],
                    run["start_frame"],
                    run["end_frame"],
                    format_time(run["start_frame"] / fps),
                    format_time((run["end_frame"] + 1) / fps),
                    round((run["end_frame"] - run["start_frame"] + 1) / fps, 3),
                ]
            )
    return labels_path, transitions_path, runs_path


def write_clip_manifest(output_dir: Path, stem: str, clip_rows: list[dict]):
    path = output_dir / f"{stem}_run_clips.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["run_id", "label", "start_frame", "end_frame", "start_timecode", "end_timecode", "clip"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clip_rows)
    return path


def write_html_report(
    output_dir,
    stem,
    video_path,
    storyboard_path,
    labels_path,
    transitions_path,
    runs_path,
    stats,
    clips_path=None,
    clip_rows=None,
):
    report = output_dir / f"{stem}_frame_transition_report.html"
    video_src = relative_url(report, video_path)
    storyboard_src = relative_url(report, storyboard_path)
    clip_cards = ""
    if clip_rows:
        cards = []
        for row in clip_rows:
            clip_src = relative_url(report, Path(row["clip"]))
            cards.append(
                f"""
                <article class="clip-card {html.escape(row['label'])}">
                  <video autoplay muted loop playsinline preload="auto" src="{clip_src}"></video>
                  <div class="clip-meta">
                    <strong>Run {html.escape(str(row['run_id']))} · {html.escape(row['label'])}</strong>
                    <span>{html.escape(row['start_timecode'])} - {html.escape(row['end_timecode'])}</span>
                    <span>frames {html.escape(str(row['start_frame']))}-{html.escape(str(row['end_frame']))}</span>
                  </div>
                </article>
                """
            )
        clip_cards = "\n".join(cards)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(stem)} frame transition labels</title>
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; background:#f6f7f9; color:#20252b; }}
    header, main {{ max-width:1280px; margin:0 auto; padding:20px 24px; }}
    header {{ background:#fff; border-bottom:1px solid #dde2e7; max-width:none; }}
    h1 {{ margin:0 0 12px; font-size:22px; }}
    .metrics {{ display:flex; flex-wrap:wrap; gap:10px; }}
    .metric {{ background:#fff; border:1px solid #dde2e7; padding:8px 10px; font-size:13px; }}
    .metric b {{ display:block; font-size:17px; }}
    video {{ width:100%; max-height:520px; background:#111; border:1px solid #dde2e7; }}
    img {{ width:100%; display:block; border:1px solid #dde2e7; }}
    .panel {{ background:#fff; border:1px solid #dde2e7; padding:14px; margin:14px 0; }}
    a {{ color:#2457c5; }}
    .clip-grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap:14px; margin-top:14px; }}
    .clip-card {{ background:#fff; border:1px solid #dde2e7; }}
    .clip-card.transition {{ border-color:#c43e3a; }}
    .clip-card.normal {{ border-color:#36a676; }}
    .clip-card video {{ width:100%; aspect-ratio:16/9; object-fit:cover; border:0; display:block; }}
    .clip-meta {{ padding:10px 12px 12px; display:flex; flex-direction:column; gap:4px; color:#66707a; font-size:13px; }}
    .clip-meta strong {{ color:#20252b; font-size:15px; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(stem)} frame transition labels</h1>
    <div class="metrics">
      <div class="metric"><b>{stats['frame_count']}</b>frames</div>
      <div class="metric"><b>{stats['transition_frame_count']}</b>transition frames</div>
      <div class="metric"><b>{stats['transition_count']}</b>transition regions</div>
      <div class="metric"><b>{stats['normal_run_count']}</b>normal runs</div>
      <div class="metric"><b>{stats['fps']:.3f}</b>fps</div>
    </div>
  </header>
  <main>
    <section class="panel"><video controls preload="metadata" src="{video_src}"></video></section>
    {'<section class="clip-grid">' + clip_cards + '</section>' if clip_cards else ''}
  </main>
</body>
</html>
"""
    report.write_text(html_text, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Create frame-level normal/transition labels and collapsed storyboard.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_frame_labels"))
    parser.add_argument("--low-threshold", type=float, default=0.18)
    parser.add_argument("--peak-threshold", type=float, default=0.38)
    parser.add_argument("--merge-gap", type=int, default=2)
    parser.add_argument("--pad-frames", type=int, default=1)
    parser.add_argument("--max-transition-frames", type=int, default=24)
    parser.add_argument("--method", choices=["anchored", "continuous", "rerank"], default="anchored")
    parser.add_argument("--model-peak-threshold", type=float, default=0.5)
    parser.add_argument("--weak-model-threshold", type=float, default=0.25)
    parser.add_argument("--visual-peak-threshold", type=float, default=0.62)
    parser.add_argument("--expansion-threshold", type=float, default=0.18)
    parser.add_argument("--hard-pad-frames", type=int, default=2)
    parser.add_argument("--max-expand-frames", type=int, default=12)
    parser.add_argument("--candidate-threshold", type=float, default=0.18)
    parser.add_argument("--accept-threshold", type=float, default=0.46)
    parser.add_argument("--prepost-threshold", type=float, default=0.20)
    parser.add_argument("--rerank-window", type=int, default=10)
    parser.add_argument("--rerank-guard", type=int, default=2)
    parser.add_argument("--nms-radius", type=int, default=6)
    parser.add_argument("--gradual-expand-threshold", type=float, default=0.20)
    parser.add_argument("--export-clips", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    frames, thumbs, smalls, fps, source_width, source_height = read_video(args.video)
    single, many = load_predictions(args.predictions, len(frames))
    model = np.maximum(single, many)
    changes = compute_visual_change(smalls)
    visual = changes["visual_score"]
    if args.method == "rerank":
        labels, transitions, fused = label_transition_frames_rerank(
            smalls,
            model,
            many,
            visual,
            candidate_threshold=args.candidate_threshold,
            accept_threshold=args.accept_threshold,
            prepost_threshold=args.prepost_threshold,
            window=args.rerank_window,
            guard=args.rerank_guard,
            nms_radius=args.nms_radius,
            hard_pad_frames=args.hard_pad_frames,
            gradual_expand_threshold=args.gradual_expand_threshold,
            max_expand_frames=args.max_expand_frames,
        )
    elif args.method == "anchored":
        labels, transitions, fused = label_transition_frames_anchored(
            model,
            many,
            visual,
            model_peak_threshold=args.model_peak_threshold,
            weak_model_threshold=args.weak_model_threshold,
            visual_peak_threshold=args.visual_peak_threshold,
            expansion_threshold=args.expansion_threshold,
            merge_gap=args.merge_gap,
            hard_pad_frames=args.hard_pad_frames,
            max_expand_frames=args.max_expand_frames,
        )
    else:
        labels, transitions, fused = label_transition_frames(
            model,
            many,
            visual,
            low_threshold=args.low_threshold,
            peak_threshold=args.peak_threshold,
            merge_gap=args.merge_gap,
            pad_frames=args.pad_frames,
        )
    runs = build_runs(labels)

    storyboard_path = args.output_dir / f"{stem}_collapsed_storyboard.jpg"
    save_storyboard(storyboard_path, thumbs, runs, fps, fused, args.max_transition_frames)
    labels_path, transitions_path, runs_path = write_csvs(args.output_dir, stem, labels, transitions, runs, fps, model, many, visual, fused)
    clips_path = None
    clip_rows = None
    if args.export_clips:
        clips_dir, clip_rows = export_run_clips(args.video, args.output_dir, stem, runs, fps)
        clips_path = write_clip_manifest(args.output_dir, stem, clip_rows)

    stats = {
        "frame_count": len(frames),
        "transition_frame_count": int(labels.sum()),
        "transition_count": len(transitions),
        "normal_run_count": sum(1 for run in runs if run["label"] == "normal"),
        "fps": fps,
        "source_width": source_width,
        "source_height": source_height,
    }
    stats_path = args.output_dir / f"{stem}_frame_label_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
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

    print(f"frames={len(frames)} transition_regions={len(transitions)} transition_frames={int(labels.sum())}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {storyboard_path}")
    print(f"Wrote: {labels_path}")
    print(f"Wrote: {transitions_path}")
    print(f"Wrote: {runs_path}")
    if clips_path:
        print(f"Wrote: {clips_path}")


if __name__ == "__main__":
    main()
