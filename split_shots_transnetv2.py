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

from transnetv2_pytorch import TransNetV2  # noqa: E402


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


def write_outputs(
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


def main():
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
    args = parser.parse_args()

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
    csv_path, json_path, pred_path, rows = write_outputs(
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


if __name__ == "__main__":
    main()
