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


def main():
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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
