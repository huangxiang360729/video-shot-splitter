"""Local video shot and transition splitting pipeline.

This file contains the full local processing code used by app.py. It exposes
three CLI stages so the web app can show progress while still keeping the video
analysis code in one normal, readable module:

    python video_splitter.py run VIDEO [args]
    python video_splitter.py transnet VIDEO [args]
    python video_splitter.py autoshot VIDEO [args]
    python video_splitter.py sweep VIDEO [args]
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
TRANSNET_PYTORCH_DIR = ROOT / ".transnetv2" / "inference-pytorch"
if str(TRANSNET_PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSNET_PYTORCH_DIR))

try:
    from transnetv2_pytorch import TransNetV2  # noqa: E402
except Exception:  # TransNet is only required when running the transnet stage.
    TransNetV2 = None


# ---------------------------------------------------------------------------
# TransNetV2 stage

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


def draw_label_bar(image: np.ndarray, label: str, sublabel: str | None = None) -> np.ndarray:
    bar_height = 42 if sublabel else 30
    canvas = np.full((image.shape[0] + bar_height, image.shape[1], 3), 246, dtype=np.uint8)
    canvas[: image.shape[0]] = image
    cv2.putText(canvas, label, (10, image.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (28, 32, 36), 1, cv2.LINE_AA)
    if sublabel:
        cv2.putText(
            canvas,
            sublabel,
            (10, image.shape[0] + 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (92, 98, 105),
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_shot_thumbnail(video_path: Path, row: dict, fps: float, out_path: Path):
    frame_ids = [
        int(row["start_frame"]),
        (int(row["start_frame"]) + int(row["end_frame"])) // 2,
        int(row["end_frame"]),
    ]
    labels = ["start", "middle", "end"]
    panels = []
    for label, frame_id in zip(labels, frame_ids):
        seconds = frame_id / fps
        frame = read_source_frame(video_path, frame_id)
        panels.append(draw_label_bar(frame, label, f"f{frame_id}  {format_time(seconds)}"))

    gap = np.full((panels[0].shape[0], 8, 3), 238, dtype=np.uint8)
    image = np.hstack([panels[0], gap, panels[1], gap, panels[2]])
    cv2.imwrite(str(out_path), image)


def save_cut_pair(
    video_path: Path,
    before_frame: int,
    after_frame: int,
    fps: float,
    score: float,
    out_path: Path,
):
    before = draw_label_bar(
        read_source_frame(video_path, before_frame, target_width=430),
        "before",
        f"f{before_frame}  {format_time(before_frame / fps)}",
    )
    after = draw_label_bar(
        read_source_frame(video_path, after_frame, target_width=430),
        "after",
        f"f{after_frame}  {format_time(after_frame / fps)}",
    )
    gap = np.full((before.shape[0], 10, 3), 235, dtype=np.uint8)
    image = np.hstack([before, gap, after])
    header = np.full((54, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        f"cut at {format_time(after_frame / fps)}   score={score:.3f}",
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (24, 28, 32),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), np.vstack([header, image]))


def save_transition_strip(video_path: Path, transition: dict, fps: float, out_path: Path):
    frame_ids = [
        int(transition["start_frame"]),
        int(transition["peak_frame"]),
        int(transition["end_frame"]),
    ]
    labels = ["transition start", "peak", "transition end"]
    panels = []
    for label, frame_id in zip(labels, frame_ids):
        panels.append(
            draw_label_bar(
                read_source_frame(video_path, frame_id, target_width=360),
                label,
                f"f{frame_id}  {format_time(frame_id / fps)}",
            )
        )
    gap = np.full((panels[0].shape[0], 8, 3), 238, dtype=np.uint8)
    image = np.hstack([panels[0], gap, panels[1], gap, panels[2]])
    header = np.full((50, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        f"{transition['transition_type']}   score={transition['peak_score']:.3f}   frames {transition['start_frame']}-{transition['end_frame']}",
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (24, 28, 32),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), np.vstack([header, image]))


def draw_score_plot(
    out_path: Path,
    single: np.ndarray,
    many: np.ndarray,
    selected: np.ndarray,
    rows: list[dict],
    fps: float,
    threshold: float,
):
    width, height = 1400, 360
    left, right, top, bottom = 64, 24, 24, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (left, top), (left + plot_w, top + plot_h), (225, 230, 235), 1)

    for value in [0.25, 0.5, 0.75]:
        y = top + int(round((1.0 - value) * plot_h))
        cv2.line(image, (left, y), (left + plot_w, y), (238, 241, 244), 1)
        cv2.putText(image, f"{value:.2f}", (16, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 118, 126), 1)

    threshold_y = top + int(round((1.0 - threshold) * plot_h))
    cv2.line(image, (left, threshold_y), (left + plot_w, threshold_y), (38, 92, 210), 2)
    cv2.putText(
        image,
        f"threshold {threshold:.2f}",
        (left + 8, threshold_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (38, 92, 210),
        1,
        cv2.LINE_AA,
    )

    def point(frame_idx: int, score: float):
        x = left + int(round(frame_idx * plot_w / max(1, len(single) - 1)))
        y = top + int(round((1.0 - float(np.clip(score, 0, 1))) * plot_h))
        return x, y

    def draw_series(values: np.ndarray, color: tuple[int, int, int], thickness: int):
        pts = np.array([point(i, v) for i, v in enumerate(values)], dtype=np.int32)
        cv2.polylines(image, [pts], False, color, thickness, cv2.LINE_AA)

    draw_series(many, (61, 139, 91), 2)
    draw_series(single, (226, 128, 38), 2)
    draw_series(selected, (35, 38, 42), 2)

    for row in rows[1:]:
        frame = int(row["start_frame"])
        x, _ = point(frame, 0)
        cv2.line(image, (x, top), (x, top + plot_h), (190, 58, 54), 1)

    total_seconds = len(single) / fps
    for sec in np.linspace(0, total_seconds, 6):
        x = left + int(round((sec * fps) * plot_w / max(1, len(single) - 1)))
        cv2.line(image, (x, top + plot_h), (x, top + plot_h + 6), (120, 126, 132), 1)
        cv2.putText(image, f"{sec:.1f}s", (x - 18, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 76, 82), 1)

    legend_x = width - 420
    for label, color, offset in [
        ("selected score", (35, 38, 42), 0),
        ("single", (226, 128, 38), 120),
        ("many_hot", (61, 139, 91), 200),
        ("cuts", (190, 58, 54), 300),
    ]:
        cv2.line(image, (legend_x + offset, 28), (legend_x + offset + 24, 28), color, 3)
        cv2.putText(image, label, (legend_x + offset + 30, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (45, 50, 55), 1)

    cv2.imwrite(str(out_path), image)


def draw_timeline(out_path: Path, rows: list[dict], frame_count: int):
    width, height = 1400, 118
    left, right, top = 44, 24, 34
    bar_w = width - left - right
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    colors = [
        (66, 135, 245),
        (54, 166, 118),
        (229, 154, 57),
        (196, 91, 86),
        (132, 100, 208),
        (48, 153, 173),
    ]
    for idx, row in enumerate(rows):
        x1 = left + int(round(int(row["start_frame"]) * bar_w / max(1, frame_count - 1)))
        x2 = left + int(round((int(row["end_frame"]) + 1) * bar_w / max(1, frame_count - 1)))
        color = colors[idx % len(colors)]
        cv2.rectangle(image, (x1, top), (max(x1 + 2, x2), top + 32), color, -1)
        if x2 - x1 > 42:
            cv2.putText(image, str(row["shot_id"]), (x1 + 8, top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    cv2.putText(image, "Shot timeline", (left, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (32, 36, 40), 1, cv2.LINE_AA)
    cv2.putText(image, "duration is proportional to width", (left, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (96, 104, 112), 1)
    cv2.imwrite(str(out_path), image)


def relative_url(from_file: Path, target: Path) -> str:
    rel = os.path.relpath(target, from_file.parent).replace(os.sep, "/")
    return quote(rel, safe="/:._-")


def write_visual_report(
    output_dir: Path,
    video_path: Path,
    rows: list[dict],
    single: np.ndarray,
    many: np.ndarray,
    selected: np.ndarray,
    fps: float,
    width: int,
    height: int,
    threshold: float,
    score_mode: str,
    min_shot_seconds: float,
    transitions: list[dict] | None = None,
) -> Path:
    assets = output_dir / f"{video_path.stem}_report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{video_path.stem}_report.html"

    timeline_path = assets / "timeline.jpg"
    score_path = assets / "score_plot.jpg"
    draw_timeline(timeline_path, rows, len(single))
    draw_score_plot(score_path, single, many, selected, rows, fps, threshold)

    shot_cards = []
    for row in rows:
        thumb_path = assets / f"shot_{int(row['shot_id']):03d}.jpg"
        save_shot_thumbnail(video_path, row, fps, thumb_path)
        shot_cards.append((row, thumb_path))

    cut_cards = []
    for index, row in enumerate(rows[1:], start=1):
        cut_frame = int(row["start_frame"])
        before = max(0, cut_frame - 2)
        after = min(len(selected) - 1, cut_frame + 2)
        score = float(selected[min(len(selected) - 1, cut_frame)])
        cut_path = assets / f"cut_{index:03d}.jpg"
        save_cut_pair(video_path, before, after, fps, score, cut_path)
        cut_cards.append((row, cut_path, score))

    transition_cards = []
    for transition in transitions or []:
        path = assets / f"transition_{int(transition['transition_id']):03d}.jpg"
        save_transition_strip(video_path, transition, fps, path)
        transition_cards.append((transition, path))

    video_src = relative_url(report_path, video_path)
    timeline_src = relative_url(report_path, timeline_path)
    score_src = relative_url(report_path, score_path)
    title = html.escape(video_path.name)
    total_seconds = len(single) / fps

    cut_html = "\n".join(
        f"""
        <article class="cut-card">
          <img src="{relative_url(report_path, path)}" alt="cut before and after">
          <div class="meta">
            <strong>Cut before shot {html.escape(str(row['shot_id']))}</strong>
            <span>{html.escape(row['start_timecode'])} · score {score:.3f}</span>
          </div>
        </article>
        """
        for row, path, score in cut_cards
    )
    shot_html = "\n".join(
        f"""
        <article class="shot-card">
          <img src="{relative_url(report_path, path)}" alt="shot {html.escape(str(row['shot_id']))}">
          <div class="meta">
            <strong>Shot {html.escape(str(row['shot_id']))}</strong>
            <span>{html.escape(row['start_timecode'])} - {html.escape(row['end_timecode'])}</span>
            <span>{html.escape(str(row['duration_seconds']))}s · frames {html.escape(str(row['start_frame']))}-{html.escape(str(row['end_frame']))}</span>
          </div>
        </article>
        """
        for row, path in shot_cards
    )
    transition_html = "\n".join(
        f"""
        <article class="shot-card">
          <img src="{relative_url(report_path, path)}" alt="transition {html.escape(str(item['transition_id']))}">
          <div class="meta">
            <strong>Transition {html.escape(str(item['transition_id']))} · {html.escape(item['transition_type'])}</strong>
            <span>{html.escape(format_time(item['start_frame'] / fps))} - {html.escape(format_time((item['end_frame'] + 1) / fps))}</span>
            <span>peak f{html.escape(str(item['peak_frame']))} · score {html.escape(str(item['peak_score']))} · frames {html.escape(str(item['start_frame']))}-{html.escape(str(item['end_frame']))}</span>
          </div>
        </article>
        """
        for item, path in transition_cards
    )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} shot report</title>
  <style>
    :root {{ color-scheme: light; --line:#dde2e7; --text:#20252b; --muted:#66707a; --bg:#f6f7f9; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; color:var(--text); background:var(--bg); }}
    header {{ padding:24px 28px 16px; background:#fff; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 12px; font-size:22px; font-weight:650; letter-spacing:0; }}
    h2 {{ margin:28px 0 12px; font-size:18px; font-weight:650; }}
    main {{ padding:20px 28px 40px; max-width:1480px; margin:0 auto; }}
    video {{ width:100%; max-height:520px; background:#111; display:block; border:1px solid var(--line); }}
    .metrics {{ display:flex; flex-wrap:wrap; gap:10px; }}
    .metric {{ padding:8px 10px; border:1px solid var(--line); background:#fff; font-size:13px; }}
    .metric b {{ display:block; font-size:16px; margin-bottom:2px; }}
    .panel {{ background:#fff; border:1px solid var(--line); padding:14px; margin-top:14px; }}
    .wide-img {{ width:100%; display:block; border:1px solid var(--line); }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap:14px; }}
    .shot-grid {{ grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); }}
    article {{ background:#fff; border:1px solid var(--line); }}
    article img {{ width:100%; display:block; }}
    .meta {{ padding:10px 12px 12px; display:flex; gap:5px; flex-direction:column; font-size:13px; color:var(--muted); }}
    .meta strong {{ color:var(--text); font-size:15px; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; font-size:13px; }}
    th, td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
    th {{ color:#48515a; background:#f0f2f5; font-weight:650; }}
    @media (max-width: 720px) {{
      header, main {{ padding-left:14px; padding-right:14px; }}
      .grid, .shot-grid {{ grid-template-columns: 1fr; }}
      th, td {{ white-space:normal; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="metrics">
      <div class="metric"><b>{len(rows)}</b>shots</div>
      <div class="metric"><b>{len(transitions or [])}</b>transitions</div>
      <div class="metric"><b>{len(single)}</b>frames</div>
      <div class="metric"><b>{fps:.3f}</b>fps</div>
      <div class="metric"><b>{total_seconds:.3f}s</b>duration</div>
      <div class="metric"><b>{html.escape(score_mode)}</b>score mode</div>
      <div class="metric"><b>{threshold:.2f}</b>threshold</div>
      <div class="metric"><b>{min_shot_seconds:.2f}s</b>min shot</div>
      <div class="metric"><b>{width}x{height}</b>source</div>
    </div>
  </header>
  <main>
    <section class="panel"><video controls preload="metadata" src="{video_src}"></video></section>
    <h2>Timeline</h2>
    <section class="panel"><img class="wide-img" src="{timeline_src}" alt="shot timeline"></section>
    <h2>Confidence</h2>
    <section class="panel"><img class="wide-img" src="{score_src}" alt="score plot"></section>
    <h2>Transition Frames</h2>
    <section class="grid shot-grid">{transition_html}</section>
    <h2>Cut Review</h2>
    <section class="grid">{cut_html}</section>
    <h2>Shot Review</h2>
    <section class="grid shot-grid">{shot_html}</section>
    <h2>Shot Table</h2>
    <section class="panel">
      <table>
        <thead><tr><th>#</th><th>Start</th><th>End</th><th>Duration</th><th>Frames</th></tr></thead>
        <tbody>
          {''.join(f"<tr><td>{r['shot_id']}</td><td>{html.escape(r['start_timecode'])}</td><td>{html.escape(r['end_timecode'])}</td><td>{r['duration_seconds']}s</td><td>{r['start_frame']}-{r['end_frame']}</td></tr>" for r in rows)}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def export_clips(video_path: Path, rows: list[dict], clips_dir: Path, ffmpeg_path: str | None, clip_mode: str):
    if ffmpeg_path is None:
        try:
            import imageio_ffmpeg

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError("FFmpeg was not found; install ffmpeg or imageio_ffmpeg to export clips.") from exc

    clips_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        out = clips_dir / f"shot_{row['shot_id']:03d}_{row['start_timecode'].replace(':', '-')}.mp4"
        cmd = [
            ffmpeg_path,
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


def run_transnet_cli(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Split a video into shots with TransNetV2.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "transnetv2-pytorch-weights.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
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

    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Reading video: {args.video}")
    frames, fps, width, height = read_video_frames(args.video)
    print(f"Decoded {len(frames)} frames, fps={fps:.3f}, source={width}x{height}")
    print(f"Running TransNetV2 on {device}")
    single, many = predict_transnet(frames, args.weights, device)
    scores = select_scores(single, many, args.score_mode)
    transition_low = args.transition_low_threshold if args.transition_low_threshold is not None else args.threshold
    transition_peak = args.transition_peak_threshold if args.transition_peak_threshold is not None else args.threshold
    transitions = detect_transition_regions(
        scores,
        single,
        many,
        low_threshold=transition_low,
        peak_threshold=transition_peak,
        merge_gap=args.transition_merge_gap,
    )
    if args.use_transition_boundaries:
        scenes = scenes_from_transitions(transitions, len(frames))
    else:
        scenes = predictions_to_scenes(scores, args.threshold)
    scenes = merge_short_scenes(scenes, int(round(args.min_shot_seconds * fps)))
    csv_path, json_path, pred_path, rows = write_transnet_outputs(
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
    transition_path = write_transitions(args.output_dir, args.video, transitions, fps, transition_low, transition_peak)
    if args.visualize_report:
        report_path = write_visual_report(
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
        export_clips(args.video, rows, clips_dir, ffmpeg_path=None, clip_mode=args.clip_mode)
        print(f"Wrote clips: {clips_dir}")
    print(f"Detected {len(scenes)} shots")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {pred_path}")
    print(f"Wrote: {transition_path}")



# ---------------------------------------------------------------------------
# AutoShot stage


def write_autoshot_predictions(path: Path, scores: list[float], fps: float) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "seconds", "single_frame_score", "many_hot_score"])
        for idx, score in enumerate(scores):
            writer.writerow([idx, round(idx / fps, 3), float(score), float(score)])


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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ShotSplitter(device=args.device) as splitter:
        analysis = splitter.analyze(
            args.video,
            threshold=args.threshold,
            include_transition_frames=False,
            include_scores=True,
        )

    _, _, fps = read_video(args.video)
    scores = np.asarray(analysis["frame_scores"], dtype=np.float32)
    predictions_path = args.output_dir / f"{args.video.stem}_autoshot_frame_predictions.csv"
    write_autoshot_predictions(predictions_path, scores.tolist(), fps)

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


# ---------------------------------------------------------------------------
# Adaptive candidate sweep stage

def format_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def read_video(video_path: Path, size=(160, 90)):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    smalls = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        smalls.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise RuntimeError("No frames decoded.")
    return frames, np.asarray(smalls), float(fps)


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


def rel(from_file, target):
    return quote(os.path.relpath(target, from_file.parent).replace(os.sep, "/"), safe="/:._-")


def write_outputs(out_dir, video_path, candidates, clip_rows, fps, frame_count, threshold):
    csv_path = out_dir / "candidate_sweep.csv"
    fields = [
        "candidate_id",
        "peak_frame",
        "peak_timecode",
        "peak_seconds",
        "score",
        "transnet_score",
        "autoshot_score",
        "hist_score",
        "gray_score",
        "edge_score",
        "flash_score",
        "prepost_score",
        "visual_supports",
        "transition_kind",
        "reason",
        "clip_start",
        "clip_end",
        "clip",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in clip_rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    report = out_dir / "candidate_sweep_report.html"
    cards = []
    for row in clip_rows:
        cards.append(
            f"""
            <article>
              <video autoplay muted loop playsinline preload="auto" src="{rel(report, Path(row['clip']))}"></video>
              <div class="meta">
                <strong>Candidate {row['candidate_id']} · {html.escape(row['peak_timecode'])}</strong>
                <span>score {row['score']} · {html.escape(row['reason'])}</span>
                <span>T {row['transnet_score']} · A {row['autoshot_score']} · H {row['hist_score']} · F {row['flash_score']} · P {row['prepost_score']}</span>
              </div>
            </article>
            """
        )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Candidate sweep</title>
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; background:#f6f7f9; color:#20252b; }}
    header, main {{ max-width:1360px; margin:0 auto; padding:20px 24px; }}
    header {{ background:#fff; border-bottom:1px solid #dde2e7; max-width:none; }}
    h1 {{ margin:0 0 12px; font-size:22px; }}
    .metrics {{ display:flex; gap:10px; flex-wrap:wrap; }}
    .metric {{ background:#fff; border:1px solid #dde2e7; padding:8px 10px; font-size:13px; }}
    .metric b {{ display:block; font-size:17px; }}
    .panel {{ background:#fff; border:1px solid #dde2e7; padding:14px; margin:14px 0; }}
    .source {{ width:100%; max-height:520px; background:#111; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:14px; }}
    article {{ background:#fff; border:1px solid #dde2e7; }}
    article video {{ width:100%; aspect-ratio:16/9; object-fit:cover; display:block; }}
    .meta {{ padding:10px 12px 12px; display:flex; flex-direction:column; gap:4px; font-size:13px; color:#66707a; }}
    .meta strong {{ color:#20252b; font-size:15px; }}
  </style>
</head>
<body>
  <header>
    <h1>Candidate sweep</h1>
    <div class="metrics">
      <div class="metric"><b>{len(candidates)}</b>candidates</div>
      <div class="metric"><b>{frame_count}</b>frames</div>
      <div class="metric"><b>{fps:.3f}</b>fps</div>
      <div class="metric"><b>{threshold:.2f}</b>threshold</div>
    </div>
  </header>
  <main>
    <section class="panel"><video class="source" controls preload="metadata" src="{rel(report, video_path)}"></video></section>
    <section class="grid">{''.join(cards)}</section>
  </main>
</body>
</html>
"""
    report.write_text(html_text, encoding="utf-8")
    meta = {
        "video": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "threshold": threshold,
        "candidate_count": len(candidates),
    }
    (out_dir / "candidate_sweep_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, csv_path


def write_run_report(out_dir, video_path, run_rows, fps, frame_count, threshold):
    csv_path = out_dir / "normal_transition_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["run_id", "label", "start_frame", "end_frame", "start_timecode", "end_timecode", "duration_seconds", "source", "clip"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(run_rows)

    report = out_dir / "normal_transition_report.html"

    def cards(label):
        html_cards = []
        for row in [r for r in run_rows if r["label"] == label]:
            html_cards.append(
                f"""
                <article class="{label}">
                  <video autoplay muted loop playsinline preload="auto" src="{rel(report, Path(row['clip']))}"></video>
                  <div class="meta">
                    <strong>Run {row['run_id']} · {label}</strong>
                    <span>{html.escape(row['start_timecode'])} - {html.escape(row['end_timecode'])}</span>
                    <span>{row['duration_seconds']}s · frames {row['start_frame']}-{row['end_frame']}</span>
                    <span>{html.escape(row.get('source', ''))}</span>
                  </div>
                </article>
                """
            )
        return "".join(html_cards)

    normal_count = sum(1 for row in run_rows if row["label"] == "normal")
    transition_count = sum(1 for row in run_rows if row["label"] == "transition")
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Normal / Transition clips</title>
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; background:#f6f7f9; color:#20252b; }}
    header, main {{ max-width:1360px; margin:0 auto; padding:20px 24px; }}
    header {{ background:#fff; border-bottom:1px solid #dde2e7; max-width:none; }}
    h1 {{ margin:0 0 12px; font-size:22px; }}
    h2 {{ margin:24px 0 12px; font-size:18px; }}
    .metrics {{ display:flex; gap:10px; flex-wrap:wrap; }}
    .metric {{ background:#fff; border:1px solid #dde2e7; padding:8px 10px; font-size:13px; }}
    .metric b {{ display:block; font-size:17px; }}
    .panel {{ background:#fff; border:1px solid #dde2e7; padding:14px; margin:14px 0; }}
    .source {{ width:100%; max-height:520px; background:#111; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:14px; }}
    article {{ background:#fff; border:1px solid #dde2e7; }}
    article.normal {{ border-color:#36a676; }}
    article.transition {{ border-color:#c43e3a; }}
    article video {{ width:100%; aspect-ratio:16/9; object-fit:cover; display:block; }}
    .meta {{ padding:10px 12px 12px; display:flex; flex-direction:column; gap:4px; font-size:13px; color:#66707a; }}
    .meta strong {{ color:#20252b; font-size:15px; }}
  </style>
</head>
<body>
  <header>
    <h1>Normal / Transition clips</h1>
    <div class="metrics">
      <div class="metric"><b>{normal_count}</b>normal clips</div>
      <div class="metric"><b>{transition_count}</b>transition clips</div>
      <div class="metric"><b>{frame_count}</b>frames</div>
      <div class="metric"><b>{fps:.3f}</b>fps</div>
      <div class="metric"><b>{threshold:.2f}</b>candidate threshold</div>
    </div>
  </header>
  <main>
    <section class="panel"><video class="source" controls preload="metadata" src="{rel(report, video_path)}"></video></section>
    <h2>Transition Clips</h2>
    <section class="grid">{cards('transition')}</section>
    <h2>Normal Clips</h2>
    <section class="grid">{cards('normal')}</section>
  </main>
  <script>
    const observer = new IntersectionObserver((entries) => {{
      for (const entry of entries) {{
        const video = entry.target;
        if (entry.isIntersecting) {{
          video.play().catch(() => {{}});
        }} else {{
          video.pause();
        }}
      }}
    }}, {{ root: null, threshold: 0.15 }});
    document.querySelectorAll('article video').forEach((video) => {{
      video.muted = true;
      video.loop = true;
      video.playsInline = true;
      observer.observe(video);
    }});
  </script>
</body>
</html>
"""
    report.write_text(html_text, encoding="utf-8")
    return report, csv_path


def run_sweep_cli(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="High-recall local candidate sweep for fast transitions.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--transnet-predictions", type=Path, default=None)
    parser.add_argument("--autoshot-predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_candidate_sweep"))
    parser.add_argument("--threshold", type=float, default=0.34)
    parser.add_argument("--model-keep-threshold", type=float, default=0.30)
    parser.add_argument("--visual-keep-threshold", type=float, default=0.58)
    parser.add_argument("--weak-visual-keep-threshold", type=float, default=0.46)
    parser.add_argument("--min-supports", type=int, default=2)
    parser.add_argument("--min-gap-seconds", type=float, default=0.25)
    parser.add_argument("--clip-seconds", type=float, default=0.9)
    parser.add_argument("--split-runs", action="store_true")
    parser.add_argument("--transition-seconds", type=float, default=0.45)
    parser.add_argument("--adaptive-runs", action="store_true")
    parser.add_argument("--hard-pad-frames", type=int, default=2)
    parser.add_argument("--complex-expand-threshold", type=float, default=0.45)
    parser.add_argument("--flash-expand-threshold", type=float, default=0.45)
    parser.add_argument("--max-complex-seconds", type=float, default=0.75)
    parser.add_argument("--dark-threshold", type=float, default=0.08)
    parser.add_argument("--bright-threshold", type=float, default=0.96)
    parser.add_argument("--luma-pad-frames", type=int, default=2)
    parser.add_argument("--min-normal-seconds", type=float, default=0.45)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, smalls, fps = read_video(args.video)
    transnet = load_model_scores(args.transnet_predictions, len(frames))
    autoshot = load_model_scores(args.autoshot_predictions, len(frames))
    signals = compute_signals(smalls)
    candidates, score, visual = build_candidates(
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
    clip_rows = export_candidate_clips(args.video, args.output_dir, candidates, fps, args.clip_seconds)
    report, csv_path = write_outputs(args.output_dir, args.video, candidates, clip_rows, fps, len(frames), args.threshold)
    if args.split_runs:
        runs = build_runs_from_candidates(
            candidates,
            len(frames),
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
        run_rows = export_run_clips(args.video, args.output_dir, runs, fps)
        run_report, run_csv = write_run_report(args.output_dir, args.video, run_rows, fps, len(frames), args.threshold)
        print(f"Wrote: {run_report}")
        print(f"Wrote: {run_csv}")
    print(f"candidates={len(candidates)}")
    print(f"Wrote: {report}")
    print(f"Wrote: {csv_path}")



# ---------------------------------------------------------------------------
# Unified CLI


def run_pipeline_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full local shot splitting pipeline.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_pipeline"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
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
        ]
    )

    report = result_dir / "normal_transition_report.html"
    print(f"Done. Open: {report}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Local video shot and transition splitting pipeline.")
    parser.add_argument("stage", choices=["run", "transnet", "autoshot", "sweep"], help="Pipeline stage to run.")
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


if __name__ == "__main__":
    main()
