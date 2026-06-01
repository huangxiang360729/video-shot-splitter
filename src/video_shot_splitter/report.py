"""报告与可视化模块：所有纯展示/报告函数。"""
from __future__ import annotations

import html
import os
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np


def relative_url(from_file: Path, target: Path) -> str:
    rel_path = os.path.relpath(target, from_file.parent).replace(os.sep, "/")
    return quote(rel_path, safe="/:._-")


def rel(from_file, target):
    return quote(os.path.relpath(target, from_file.parent).replace(os.sep, "/"), safe="/:._-")


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
    from video_shot_splitter.pipeline import read_source_frame, format_time
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
    from video_shot_splitter.pipeline import read_source_frame, format_time
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
    from video_shot_splitter.pipeline import read_source_frame, format_time
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
    from video_shot_splitter.pipeline import format_time
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


def write_outputs(out_dir, video_path, candidates, clip_rows, fps, frame_count, threshold):
    import csv
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

    report_path = out_dir / "candidate_sweep_report.html"
    cards = []
    for row in clip_rows:
        cards.append(
            f"""
            <article>
              <video autoplay muted loop playsinline preload="auto" src="{rel(report_path, Path(row['clip']))}"></video>
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
    <section class="panel"><video class="source" controls preload="metadata" src="{rel(report_path, video_path)}"></video></section>
    <section class="grid">{''.join(cards)}</section>
  </main>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")
    meta = {
        "video": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "threshold": threshold,
        "candidate_count": len(candidates),
    }
    import json
    (out_dir / "candidate_sweep_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path, csv_path


def write_run_report(out_dir, summary, source_video_path: Path | None = None):
    video_path = Path(source_video_path or summary["video"]["path"])
    report_path = out_dir / "normal_transition_report.html"

    def cards(clip_type):
        html_cards = []
        for row in summary["clip_groups"].get(clip_type, []):
            html_cards.append(
                f"""
                <article class="{clip_type}">
                  <video autoplay muted loop playsinline preload="auto" src="{rel(report_path, out_dir / row['clip_path'])}"></video>
                  <div class="meta">
                    <strong>Clip {row['id']} · {clip_type}</strong>
                    <span>{html.escape(row['start_timecode'])} - {html.escape(row['end_timecode'])}</span>
                    <span>{row['duration_sec']}s · frames {row['start_frame']}-{row['end_frame']}</span>
                    <span>{html.escape(row.get('source', ''))}</span>
                  </div>
                </article>
                """
            )
        return "".join(html_cards)

    normal_count = summary["counts"]["normal_clips"]
    transition_count = summary["counts"]["transition_clips"]
    frame_count = summary["video"]["frame_count"]
    fps = summary["video"]["fps"]
    threshold = summary["parameters"]["candidate_threshold"]
    summary_href = rel(report_path, out_dir / "summary.json")
    csv_link = ""
    if "runs_csv" in summary["outputs"]:
        csv_link = f"""<a href="{rel(report_path, out_dir / summary['outputs']['runs_csv'])}">Runs CSV</a>"""
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
    .links {{ display:flex; gap:12px; margin-top:12px; font-size:13px; }}
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
    <div class="links"><a href="{summary_href}">Summary JSON</a>{csv_link}</div>
  </header>
  <main>
    <section class="panel"><video class="source" controls preload="metadata" src="{rel(report_path, video_path)}"></video></section>
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
    report_path.write_text(html_text, encoding="utf-8")
    return report_path
