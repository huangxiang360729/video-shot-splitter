import html
import json
import mimetypes
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
RUNS_DIR = ROOT / "web_runs"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
LOG_TAIL_CHARS = 8000

PIPELINE = {
    "transnet": [
        "--device",
        "cpu",
        "--threshold",
        "0.1",
        "--score-mode",
        "max",
    ],
    "autoshot": [
        "--threshold",
        "0.45",
        "--transition-pad-frames",
        "2",
    ],
    "sweep": [
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
    ],
}

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def safe_name(name: str) -> str:
    chars = [char if char.isalnum() or char in "._- " else "_" for char in Path(name).name]
    return "".join(chars).strip(" .") or "uploaded_video.mp4"


def parse_multipart(body: bytes, content_type: str) -> tuple[str, bytes]:
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("Missing multipart boundary.")

    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    raw_boundary = ("--" + boundary).encode()

    for part in body.split(raw_boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue

        header_blob, sep, payload = part.partition(b"\r\n\r\n")
        if not sep:
            continue

        headers = header_blob.decode("utf-8", errors="replace")
        if 'name="video"' not in headers:
            continue

        filename = "uploaded_video"
        for piece in headers.split(";"):
            piece = piece.strip()
            if piece.startswith("filename="):
                filename = piece.split("=", 1)[1].strip().strip('"')
                break

        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        return safe_name(filename), payload

    raise ValueError("No uploaded video field found.")


def rel_url(path: Path) -> str:
    return "/" + quote(path.relative_to(ROOT).as_posix())


def set_job(job_id: str, **updates):
    with JOBS_LOCK:
        JOBS[job_id].update(updates)


def append_log(job_id: str, text: str):
    with JOBS_LOCK:
        current = JOBS[job_id].get("log_tail", "")
        JOBS[job_id]["log_tail"] = (current + text)[-LOG_TAIL_CHARS:]


def run_command(job_id: str, args: list, cwd: Path, log_file):
    command_line = "$ " + " ".join(str(arg) for arg in args) + "\n"
    log_file.write(command_line)
    log_file.flush()
    append_log(job_id, command_line)

    proc = subprocess.Popen(
        [str(arg) for arg in args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None

    for line in proc.stdout:
        log_file.write(line)
        log_file.flush()
        append_log(job_id, line)

    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(str(arg) for arg in args)}")


def process_job(job_id: str):
    with JOBS_LOCK:
        job = dict(JOBS[job_id])

    video_path = Path(job["video_path"])
    job_dir = Path(job["job_dir"])
    stem = video_path.stem
    transnet_dir = job_dir / "transnet"
    autoshot_dir = job_dir / "autoshot"
    result_dir = job_dir / "result"
    log_path = job_dir / "process.log"

    for path in [transnet_dir, autoshot_dir, result_dir]:
        path.mkdir(parents=True, exist_ok=True)

    try:
        started = time.time()
        set_job(job_id, status="running", step="TransNetV2 candidate scores", progress=12)

        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"Processing {video_path}\n")
            append_log(job_id, f"Processing {video_path}\n")

            run_command(
                job_id,
                [
                    PYTHON,
                    ROOT / "video_splitter.py",
                    "transnet",
                    video_path,
                    *PIPELINE["transnet"],
                    "--output-dir",
                    transnet_dir,
                ],
                ROOT,
                log,
            )

            set_job(job_id, step="AutoShot candidate scores", progress=42)
            run_command(
                job_id,
                [
                    PYTHON,
                    ROOT / "video_splitter.py",
                    "autoshot",
                    video_path,
                    *PIPELINE["autoshot"],
                    "--output-dir",
                    autoshot_dir,
                ],
                ROOT,
                log,
            )

            set_job(job_id, step="Adaptive sweep and clip export", progress=72)
            run_command(
                job_id,
                [
                    PYTHON,
                    ROOT / "video_splitter.py",
                    "sweep",
                    video_path,
                    "--transnet-predictions",
                    transnet_dir / f"{stem}_frame_predictions.csv",
                    "--autoshot-predictions",
                    autoshot_dir / f"{stem}_autoshot_frame_predictions.csv",
                    "--output-dir",
                    result_dir,
                    *PIPELINE["sweep"],
                ],
                ROOT,
                log,
            )

            elapsed = time.time() - started
            log.write(f"Done in {elapsed:.1f}s\n")
            append_log(job_id, f"Done in {elapsed:.1f}s\n")

        report = result_dir / "normal_transition_report.html"
        set_job(job_id, status="done", step="Complete", progress=100, report_url=rel_url(report), log_url=rel_url(log_path))
    except Exception as exc:
        append_log(job_id, "\n" + traceback.format_exc())
        set_job(job_id, status="error", step="Failed", error=str(exc), progress=100, log_url=rel_url(log_path))


def page_shell(body: str, title: str = "Video Shot Splitter") -> bytes:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ --bg:#f4f6f8; --ink:#1d2733; --muted:#657181; --line:#dce2ea; --blue:#2457c5; --red:#b42318; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ background:#fff; border-bottom:1px solid var(--line); }}
    .wrap {{ max-width:1060px; margin:0 auto; padding:24px; }}
    h1 {{ margin:0 0 6px; font-size:24px; letter-spacing:0; }}
    h2 {{ margin:0 0 12px; font-size:18px; }}
    p {{ line-height:1.55; }}
    .sub {{ margin:0; color:var(--muted); }}
    .panel {{ background:#fff; border:1px solid var(--line); padding:20px; margin-top:18px; }}
    .upload {{ border:1px dashed #aab5c2; background:#fbfcfe; padding:22px; }}
    input[type=file] {{ width:100%; padding:12px; background:#fff; border:1px solid var(--line); }}
    button {{ margin-top:14px; padding:10px 16px; border:1px solid #1f4ea8; background:var(--blue); color:#fff; cursor:pointer; font-weight:600; }}
    button:disabled {{ opacity:.6; cursor:not-allowed; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }}
    .metric {{ border:1px solid var(--line); background:#fff; padding:12px; }}
    .metric b {{ display:block; font-size:20px; }}
    .progress {{ height:12px; background:#e8edf3; overflow:hidden; }}
    .bar {{ height:100%; width:0%; background:var(--blue); transition:width .35s ease; }}
    pre {{ white-space:pre-wrap; overflow:auto; max-height:360px; background:#111827; color:#d1d5db; padding:14px; font-size:12px; }}
    .status {{ display:flex; justify-content:space-between; gap:12px; color:var(--muted); }}
    .error {{ color:var(--red); }}
    a {{ color:var(--blue); }}
  </style>
</head>
<body>
  <header><div class="wrap"><h1>Video Shot Splitter</h1><p class="sub">Local transition and normal clip analysis</p></div></header>
  <main class="wrap">{body}</main>
</body>
</html>""".encode("utf-8")


def index_page() -> bytes:
    return page_shell(
        """
        <section class="panel upload">
          <h2>上传视频</h2>
          <form id="uploadForm" method="post" action="/upload" enctype="multipart/form-data">
            <input id="videoInput" type="file" name="video" accept="video/*" required>
            <button id="submitButton" type="submit">开始分析</button>
          </form>
          <p class="sub">处理在本机完成。页面会显示实时进度，完成后自动进入结果页。</p>
        </section>
        <section id="uploadProgress" class="panel" style="display:none">
          <h2>上传中</h2>
          <div class="status"><span id="uploadName">Preparing upload</span><span id="uploadPct">0%</span></div>
          <div class="progress"><div id="uploadBar" class="bar"></div></div>
          <p id="uploadMessage" class="sub">上传完成后会自动进入分析进度页。</p>
        </section>
        <section class="panel">
          <div class="grid">
            <div class="metric"><b>1</b>TransNetV2 候选</div>
            <div class="metric"><b>2</b>AutoShot 候选</div>
            <div class="metric"><b>3</b>自适应转场扫描</div>
            <div class="metric"><b>4</b>生成 Normal / Transition clips</div>
          </div>
        </section>
        <script>
          const form = document.getElementById('uploadForm');
          const input = document.getElementById('videoInput');
          const button = document.getElementById('submitButton');
          const progress = document.getElementById('uploadProgress');
          const bar = document.getElementById('uploadBar');
          const pct = document.getElementById('uploadPct');
          const nameEl = document.getElementById('uploadName');
          const msg = document.getElementById('uploadMessage');

          form.addEventListener('submit', (event) => {
            event.preventDefault();
            if (!input.files.length) return;

            const file = input.files[0];
            const data = new FormData();
            data.append('video', file);
            progress.style.display = 'block';
            button.disabled = true;
            nameEl.textContent = file.name;
            msg.textContent = '正在上传到本机服务...';

            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/upload');
            xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
            xhr.upload.onprogress = (event) => {
              if (!event.lengthComputable) return;
              const value = Math.max(1, Math.round(event.loaded / event.total * 100));
              bar.style.width = value + '%';
              pct.textContent = value + '%';
            };
            xhr.onload = () => {
              if (xhr.status >= 200 && xhr.status < 300) {
                const payload = JSON.parse(xhr.responseText);
                msg.textContent = '上传完成，正在进入分析进度页...';
                location.href = payload.job_url;
              } else {
                msg.innerHTML = '<span class="error">上传失败：' + xhr.responseText + '</span>';
                button.disabled = false;
              }
            };
            xhr.onerror = () => {
              msg.innerHTML = '<span class="error">上传失败：网络或本地服务错误</span>';
              button.disabled = false;
            };
            xhr.send(data);
          });
        </script>
        """
    )


def progress_page(job_id: str) -> bytes:
    return page_shell(
        f"""
        <section class="panel">
          <h2>分析进度</h2>
          <div class="status"><span id="step">Preparing</span><span id="pct">0%</span></div>
          <div class="progress"><div id="bar" class="bar"></div></div>
          <p id="message" class="sub">任务 ID: {html.escape(job_id)}</p>
        </section>
        <section class="panel">
          <h2>后台日志</h2>
          <pre id="log"></pre>
        </section>
        <script>
          async function poll() {{
            const res = await fetch('/api/jobs/{job_id}');
            const data = await res.json();
            document.getElementById('step').textContent = data.step || data.status;
            document.getElementById('pct').textContent = (data.progress || 0) + '%';
            document.getElementById('bar').style.width = (data.progress || 0) + '%';
            document.getElementById('log').textContent = data.log_tail || '';
            if (data.status === 'done') {{
              document.getElementById('message').innerHTML = '完成，正在打开结果页... <a href="' + data.report_url + '">手动打开</a>';
              location.href = data.report_url;
              return;
            }}
            if (data.status === 'error') {{
              document.getElementById('message').innerHTML = '<span class="error">处理失败：' + (data.error || 'unknown error') + '</span>';
              return;
            }}
            setTimeout(poll, 1200);
          }}
          poll();
        </script>
        """,
        "Processing",
    )


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, data: bytes, content_type: str = "text/html; charset=utf-8", status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(index_page())
            return
        if parsed.path.startswith("/jobs/"):
            self.send_bytes(progress_page(parsed.path.rsplit("/", 1)[-1]))
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                data = dict(JOBS.get(job_id, {"status": "missing", "step": "Missing", "progress": 100}))
            self.send_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return

        requested = (ROOT / unquote(parsed.path.lstrip("/"))).resolve()
        if not str(requested).startswith(str(ROOT)) or not requested.exists() or requested.is_dir():
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(requested.stat().st_size))
        self.end_headers()
        with requested.open("rb") as file:
            shutil.copyfileobj(file, self.wfile)

    def do_POST(self):
        if urlparse(self.path).path != "/upload":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                raise ValueError("Upload is empty or too large.")

            filename, payload = parse_multipart(self.rfile.read(length), self.headers.get("Content-Type", ""))
            job_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            job_dir = RUNS_DIR / job_id
            uploads = job_dir / "uploads"
            uploads.mkdir(parents=True, exist_ok=True)
            video_path = uploads / filename
            video_path.write_bytes(payload)

            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "queued",
                    "step": "Queued",
                    "progress": 3,
                    "job_dir": str(job_dir),
                    "video_path": str(video_path),
                    "log_tail": f"Uploaded {filename} ({len(payload) / 1024 / 1024:.1f} MB)\n",
                }

            threading.Thread(target=process_job, args=(job_id,), daemon=True).start()
            if self.headers.get("X-Requested-With") == "XMLHttpRequest":
                self.send_bytes(
                    json.dumps({"job_id": job_id, "job_url": f"/jobs/{job_id}"}, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            else:
                self.send_response(303)
                self.send_header("Location", f"/jobs/{job_id}")
                self.end_headers()
        except Exception as exc:
            body = f"<section class='panel'><h2>上传失败</h2><pre class='error'>{html.escape(str(exc))}</pre></section>"
            self.send_bytes(page_shell(body, "Error"), status=500)


def main():
    RUNS_DIR.mkdir(exist_ok=True)
    host = "127.0.0.1"
    port = 7860
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Open http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
