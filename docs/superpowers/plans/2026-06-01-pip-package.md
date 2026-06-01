# video-shot-splitter pip 打包实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把现有两个扁平脚本（`video_splitter.py` + `app.py`）重组为标准 src-layout 的 `video_shot_splitter` 包，发布到公共 PyPI，暴露 CLI / Python API / Web 三种入口，30MB 权重打进 wheel。

**Architecture:** src-layout（`src/video_shot_splitter/`）。`video_splitter.py` 按职责拆成 `pipeline.py`（分析逻辑）、`report.py`（HTML/可视化）、`cli.py`（argparse 入口）；`app.py` 移入 `web/`。权重用 `importlib.resources` 定位而非 `__file__` 拼路径。Web 服务的 subprocess 自调用从"调脚本文件"改为 `python -m video_shot_splitter`。

**Tech Stack:** Python ≥3.10、setuptools + pyproject.toml (PEP 621)、importlib.resources、torch / opencv-python / numpy / imageio-ffmpeg / shotsplit。

---

## 文件结构

```text
video-shot-splitter/
├── pyproject.toml                   # 新建：PEP 621 打包配置
├── LICENSE                          # 新建：项目顶层 MIT license
├── README.md                        # 保留
├── src/
│   └── video_shot_splitter/
│       ├── __init__.py              # 新建：导出 run_pipeline + __version__
│       ├── __main__.py              # 新建：python -m 入口 → cli.main
│       ├── cli.py                   # 由 video_splitter.py 拆出：argparse 入口
│       ├── pipeline.py              # 由 video_splitter.py 拆出：分析逻辑主体
│       ├── report.py                # 由 video_splitter.py 拆出：HTML/可视化
│       ├── resources.py            # 新建：importlib.resources 定位权重
│       ├── web/
│       │   ├── __init__.py          # 新建：空
│       │   └── app.py               # 由顶层 app.py 移入并改 subprocess 调用
│       ├── models/
│       │   └── transnetv2-pytorch-weights.pth   # 移入，打进 wheel
│       └── vendor/
│           ├── __init__.py          # 新建：空（让 vendor 成为包的一部分）
│           └── transnetv2/          # 移入，保留 MIT LICENSE
```

**拆分边界说明：**
- `pipeline.py`：`format_time`、`read_video_frames`、TransNet/AutoShot/sweep 的全部计算函数、candidate/run 构建、ffmpeg 导出、`write_outputs`/`build_summary`/`write_summary_json` 等。
- `report.py`：`draw_*`、`save_*`、`write_visual_report`、`write_run_report`、`relative_url`/`rel` 等纯展示函数。
- `cli.py`：`run_transnet_cli`、`run_autoshot_cli`、`run_sweep_cli`、`run_pipeline_cli`、`run_sweep_cli`、`run_smoke_test_cli`、`main`、`serve`（新增，启动 web）。

**注意（实现时已确认的事实）：**
- `run_pipeline_cli` 已经是**进程内**调用各 stage（`run_transnet_cli([...])`），CLI 路径无 subprocess，不需改。
- 只有 `web/app.py` 用 subprocess（为流式进度），三处调用需从 `[PYTHON, ROOT/"video_splitter.py", stage, ...]` 改为 `[sys.executable, "-m", "video_shot_splitter", stage, ...]`。
- `ffmpeg_path()` 已用 `imageio_ffmpeg.get_ffmpeg_exe()`，spec 中的"ffmpeg 适配"是 no-op，仅需保留验证，不改代码。
- vendor 的 TransNetV2 是 MIT（Copyright 2020 Tomáš Souček），合规，保留其 LICENSE 文件即可。

---

## Task 1: 创建包骨架与 src-layout 目录

**Files:**
- Create: `src/video_shot_splitter/__init__.py`
- Create: `src/video_shot_splitter/web/__init__.py`
- Create: `src/video_shot_splitter/vendor/__init__.py`

注意：现有顶层已有一个空的 `video-shot-splitter/src/` 目录（带连字符），那是无效包名，**不要用它**。本计划在仓库根下新建 `src/`（确认根目录是 `C:/Users/admin/Documents/视频分镜拆分`）。

- [ ] **Step 1: 创建目录与空 __init__**

```bash
mkdir -p src/video_shot_splitter/web src/video_shot_splitter/vendor src/video_shot_splitter/models
```

`src/video_shot_splitter/web/__init__.py` 内容：空文件。
`src/video_shot_splitter/vendor/__init__.py` 内容：空文件。

`src/video_shot_splitter/__init__.py` 内容（API 导出留到 Task 6 填，先放版本号占位的真实内容）：

```python
"""video-shot-splitter: 本地视频分镜/转场拆分工具。"""

__version__ = "0.1.0"
```

- [ ] **Step 2: 提交**

```bash
git add src/video_shot_splitter/__init__.py src/video_shot_splitter/web/__init__.py src/video_shot_splitter/vendor/__init__.py
git commit -m "chore: scaffold video_shot_splitter package skeleton"
```

---

## Task 2: 移动 vendor 与模型权重进包内

**Files:**
- Move: `vendor/transnetv2/` → `src/video_shot_splitter/vendor/transnetv2/`
- Move: `models/transnetv2-pytorch-weights.pth` → `src/video_shot_splitter/models/transnetv2-pytorch-weights.pth`

- [ ] **Step 1: 用 git mv 移动（保留历史）**

```bash
git mv vendor/transnetv2 src/video_shot_splitter/vendor/transnetv2
git mv models/transnetv2-pytorch-weights.pth src/video_shot_splitter/models/transnetv2-pytorch-weights.pth
```

- [ ] **Step 2: 删除残留的旧空目录与 pyc**

```bash
rm -rf src/video_shot_splitter/vendor/transnetv2/__pycache__ 2>/dev/null || true
rmdir vendor 2>/dev/null || true
rmdir models 2>/dev/null || true
```

- [ ] **Step 3: 确认权重就位**

Run: `ls -la src/video_shot_splitter/models/transnetv2-pytorch-weights.pth`
Expected: 文件存在，约 30508183 字节。

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "chore: move vendor and model weights into package"
```

---

## Task 3: 新建 resources.py 定位包内权重

**Files:**
- Create: `src/video_shot_splitter/resources.py`
- Test: `tests/test_resources.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_resources.py
from pathlib import Path
from video_shot_splitter.resources import default_weights_path


def test_default_weights_path_exists():
    p = default_weights_path()
    assert isinstance(p, Path)
    assert p.name == "transnetv2-pytorch-weights.pth"
    assert p.exists()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pip install -e . && pytest tests/test_resources.py -v`
Expected: FAIL（模块/函数不存在）。注意：本步需要 Task 5 的 pyproject 才能 `pip install -e .`；若尚未做 Task 5，先只确认 import 失败即可。

- [ ] **Step 3: 写实现**

```python
# src/video_shot_splitter/resources.py
"""定位打进包内的数据文件（模型权重等）。"""
from importlib.resources import files
from pathlib import Path


def default_weights_path() -> Path:
    """返回包内 TransNetV2 权重文件的真实路径。"""
    resource = files("video_shot_splitter").joinpath(
        "models/transnetv2-pytorch-weights.pth"
    )
    # importlib.resources 在普通安装下返回真实路径；用 as_file 兜底 zip 安装
    return Path(str(resource))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_resources.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/video_shot_splitter/resources.py tests/test_resources.py
git commit -m "feat: locate bundled model weights via importlib.resources"
```

---

## Task 4: 拆分 video_splitter.py 为 pipeline.py / report.py / cli.py

**Files:**
- Create: `src/video_shot_splitter/pipeline.py`
- Create: `src/video_shot_splitter/report.py`
- Create: `src/video_shot_splitter/cli.py`
- Delete: `video_splitter.py`（顶层，拆分后删除）

这是机械搬迁，**不改算法逻辑**。拆分依据见"文件结构"节的边界说明。

- [ ] **Step 1: 把展示类函数搬进 report.py**

把 `video_splitter.py` 中所有 `draw_*`、`save_*`、`write_visual_report`、`write_run_report`、`relative_url`/`rel` 等纯展示函数原样复制到 `src/video_shot_splitter/report.py`，文件顶部加它们用到的 import（`from pathlib import Path`、`import html`、numpy/cv2 等按需）。

- [ ] **Step 2: 把分析逻辑搬进 pipeline.py**

把剩余的计算函数（`format_time`、`read_video_frames`、`predict_transnet`、`build_candidates`、`expand_candidate_interval`、`build_runs_from_candidates`、`export_run_clips`、`export_candidate_clips`、`ffmpeg_path`、`write_synthetic_video`、`build_summary`、`write_summary_json` 等）搬进 `src/video_shot_splitter/pipeline.py`。

关键改动：原 `parser.add_argument("--weights", ..., default=ROOT / "models" / "transnetv2-pytorch-weights.pth")` 中的默认值改为用 `resources.default_weights_path()`：

```python
from video_shot_splitter.resources import default_weights_path
# ...
parser.add_argument("--weights", type=Path, default=None)
# 解析后：
weights = args.weights if args.weights is not None else default_weights_path()
```

pipeline.py 顶部 import report 中需要的展示函数：`from video_shot_splitter import report`。

- [ ] **Step 3: 把 argparse 入口搬进 cli.py**

把 `run_transnet_cli`、`run_autoshot_cli`、`run_sweep_cli`、`run_pipeline_cli`、`run_smoke_test_cli`、`main` 搬进 `src/video_shot_splitter/cli.py`，顶部 `from video_shot_splitter import pipeline, report`。`main` 的 stage 分发表保持不变，新增一个 `serve` 分支（Task 7 填充实现，这里先占位调用 `from video_shot_splitter.web.app import main as serve_main`）。

- [ ] **Step 4: 删除顶层旧脚本**

```bash
git rm video_splitter.py
```

- [ ] **Step 5: 冒烟验证 import 不报错（需 Task 5 的 pyproject）**

Run: `python -c "from video_shot_splitter import pipeline, report, cli"`
Expected: 无 ImportError。若此时还没做 Task 5，跳过本步，留到 Task 8 统一验证。

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "refactor: split video_splitter into pipeline/report/cli modules"
```

---

## Task 5: 编写 pyproject.toml 与 LICENSE

**Files:**
- Create: `pyproject.toml`
- Create: `LICENSE`

- [ ] **Step 1: 写 LICENSE（MIT，年份 2026）**

`LICENSE` 内容为标准 MIT 文本，版权行：`Copyright (c) 2026 huangxiang360729`。

- [ ] **Step 2: 写 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "video-shot-splitter"
dynamic = ["version"]
description = "本地视频分镜/转场拆分工具"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "huangxiang360729" }]
dependencies = [
    "numpy",
    "opencv-python",
    "torch",
    "imageio-ffmpeg",
    "shotsplit",
]

[project.urls]
Homepage = "https://github.com/huangxiang360729/video-shot-splitter"

[project.scripts]
video-shot-splitter = "video_shot_splitter.cli:main"

[tool.setuptools]
package-dir = { "" = "src" }

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.dynamic]
version = { attr = "video_shot_splitter.__version__" }

[tool.setuptools.package-data]
video_shot_splitter = [
    "models/*.pth",
    "vendor/transnetv2/*",
    "vendor/transnetv2/LICENSE",
]
```

- [ ] **Step 3: editable 安装**

Run: `pip install -e .`
Expected: 安装成功，结尾 `Successfully installed video-shot-splitter`。

- [ ] **Step 4: 提交**

```bash
git add pyproject.toml LICENSE
git commit -m "build: add pyproject.toml and MIT LICENSE"
```

---

## Task 6: 暴露 Python API 与 __main__

**Files:**
- Modify: `src/video_shot_splitter/__init__.py`
- Create: `src/video_shot_splitter/__main__.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_api.py
def test_run_pipeline_importable():
    from video_shot_splitter import run_pipeline, __version__
    assert callable(run_pipeline)
    assert isinstance(__version__, str)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_api.py -v`
Expected: FAIL（`run_pipeline` 无法从包导入）。

- [ ] **Step 3: 在 pipeline.py 增加库友好的 run_pipeline 函数**

在 `pipeline.py` 末尾新增（封装现有 CLI 的进程内三步逻辑，参数化而非 argparse）：

```python
def run_pipeline(video, output_dir="outputs_pipeline", device="cpu", debug=False):
    """库入口：对单个视频跑完整三步管线，返回 result 目录下 summary.json 的路径。"""
    from video_shot_splitter.cli import run_pipeline_cli
    argv = [str(video), "--output-dir", str(output_dir), "--device", device]
    if debug:
        argv.append("--debug")
    run_pipeline_cli(argv)
    from pathlib import Path
    return Path(output_dir) / "result" / "summary.json"
```

- [ ] **Step 4: __init__.py 导出**

```python
"""video-shot-splitter: 本地视频分镜/转场拆分工具。"""

__version__ = "0.1.0"

from video_shot_splitter.pipeline import run_pipeline

__all__ = ["run_pipeline", "__version__"]
```

- [ ] **Step 5: 写 __main__.py**

```python
from video_shot_splitter.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 运行确认通过**

Run: `pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add src/video_shot_splitter/__init__.py src/video_shot_splitter/__main__.py src/video_shot_splitter/pipeline.py tests/test_api.py
git commit -m "feat: expose run_pipeline Python API and python -m entry"
```

---

## Task 7: 移入 web/app.py 并修复 subprocess 自调用

**Files:**
- Move: `app.py` → `src/video_shot_splitter/web/app.py`
- Modify: `src/video_shot_splitter/web/app.py`（改三处 subprocess 调用 + main 暴露）
- Modify: `src/video_shot_splitter/cli.py`（serve 分支接入）

- [ ] **Step 1: git mv 移入 web 包**

```bash
git mv app.py src/video_shot_splitter/web/app.py
```

- [ ] **Step 2: 改三处 subprocess 命令数组**

`app.py` 中现有三处形如：

```python
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
```

把每处的 `PYTHON, ROOT / "video_splitter.py"` 两个元素，替换为 `sys.executable, "-m", "video_shot_splitter"`。三处分别对应 `"transnet"`、`"autoshot"`、`"sweep"` stage，其余参数保持不变。改完示例：

```python
run_command(
    job_id,
    [
        sys.executable,
        "-m",
        "video_shot_splitter",
        "transnet",
        video_path,
        *PIPELINE["transnet"],
        "--output-dir",
        transnet_dir,
    ],
    ROOT,
    log,
)
```

确认文件顶部已 `import sys`（现有 `PYTHON = Path(sys.executable)` 已 import sys，保留即可）。`ROOT`/`PYTHON` 旧变量若不再被引用可删除，但保留也无害。

- [ ] **Step 3: cli.py 的 serve 分支接入**

确认 `cli.py` 的 `main()` 中 `serve` 分支为：

```python
elif command == "serve":
    from video_shot_splitter.web.app import main as serve_main
    serve_main()
```

- [ ] **Step 4: 验证 serve 能启动（手动，Ctrl-C 退出）**

Run: `video-shot-splitter serve`
Expected: 打印 `Open http://127.0.0.1:7860`，进程不报错。确认后 Ctrl-C。

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "refactor: move web app into package and use python -m self-invocation"
```

---

## Task 8: 打包验证（smoke-test + wheel 构建）

**Files:**
- 无新文件，纯验证。

- [ ] **Step 1: 重新 editable 安装确保最新**

Run: `pip install -e .`
Expected: 成功。

- [ ] **Step 2: 跑包内 smoke-test（不依赖模型）**

Run: `video-shot-splitter smoke-test --output-dir outputs_smoke`
Expected: 结尾打印 `Smoke test passed: ...summary.json`，退出码 0。

- [ ] **Step 3: 验证 Python API 可用**

Run: `python -c "import video_shot_splitter as v; print(v.__version__); print(v.run_pipeline)"`
Expected: 打印 `0.1.0` 和一个 function 对象，无异常。

- [ ] **Step 4: 验证 python -m 入口**

Run: `python -m video_shot_splitter smoke-test --output-dir outputs_smoke2`
Expected: 同 Step 2，smoke test passed。

- [ ] **Step 5: 构建 wheel 并确认权重在包内**

```bash
pip install build
python -m build --wheel
```
Expected: `dist/video_shot_splitter-0.1.0-py3-none-any.whl` 生成。

验证权重打进 wheel：

```bash
python -c "import zipfile; z=zipfile.ZipFile([f for f in __import__('glob').glob('dist/*.whl')][0]); print([n for n in z.namelist() if n.endswith('.pth')])"
```
Expected: 输出包含 `video_shot_splitter/models/transnetv2-pytorch-weights.pth`。

- [ ] **Step 6: 清理验证产物并提交（若有 .gitignore 需要补）**

```bash
rm -rf outputs_smoke outputs_smoke2 build dist *.egg-info
```

确认 `.gitignore` 已忽略 `outputs*/`（现有已忽略），并追加 `dist/`、`build/`、`*.egg-info/`：

```bash
printf '\ndist/\nbuild/\n*.egg-info/\n' >> .gitignore
git add .gitignore
git commit -m "chore: ignore build artifacts"
```

---

## 验收标准

- `pip install -e .` 成功，`video-shot-splitter` 命令可用。
- `video-shot-splitter smoke-test` 通过（输出契约自检）。
- `import video_shot_splitter; run_pipeline` 可用。
- `python -m video_shot_splitter` 可用。
- `python -m build --wheel` 产出含 30MB 权重的 wheel。
- `video-shot-splitter serve` 能启动本地 Web 服务。
