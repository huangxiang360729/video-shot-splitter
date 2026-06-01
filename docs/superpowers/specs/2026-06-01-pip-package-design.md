# 将 video-shot-splitter 打包为 pip 包 — 设计文档

日期：2026-06-01

## 目标

把当前的本地视频分镜/转场拆分工具打包成可发布到公共 PyPI 的标准 Python 包，发布名 `video-shot-splitter`，import 名 `video_shot_splitter`。装完即用、可离线，对外暴露 CLI、Python API 和 Web 三种入口。

## 关键决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 目标受众 | 公共 PyPI | 任何人可 `pip install` |
| 包名 | `video-shot-splitter`（PyPI 已确认可用，404） | 与现有目录/git 仓一致 |
| 30MB 模型权重 | 打进 wheel（package data） | 装完即用、可离线；wheel ~30MB，在 PyPI 100MB 限额内 |
| 对外入口 | CLI + Python API + Web 全部 | 覆盖命令行、库调用、本地网页三种用法 |
| 打包工具 | setuptools + pyproject.toml (PEP 621) | 对 package-data/二进制数据和 torch 等重依赖兼容性最成熟 |
| 目录布局 | src-layout | 现代 PyPI 推荐做法，避免本地目录意外被 import |

## 包结构

```text
video-shot-splitter/                 # 仓库根（现有 git 仓）
├── pyproject.toml                   # 新增：打包配置（PEP 621）
├── README.md                        # 复用现有
├── LICENSE                          # 新增：项目顶层 license
├── src/
│   └── video_shot_splitter/
│       ├── __init__.py              # 导出 Python API + __version__
│       ├── __main__.py              # 支持 python -m video_shot_splitter
│       ├── cli.py                   # ← 现 video_splitter.py 的 main()/各 *_cli()
│       ├── pipeline.py              # ← 核心分析逻辑（predict/sweep/build_runs...）
│       ├── report.py                # ← HTML/可视化报告相关函数
│       ├── web/
│       │   └── app.py               # ← 现 app.py（Web 服务）
│       ├── models/
│       │   └── transnetv2-pytorch-weights.pth   # 30MB 权重，打进 wheel
│       └── vendor/
│           └── transnetv2/          # ← 现 vendor/，保留 LICENSE
```

要点：
- 采用 src-layout（`src/` 下）。现有空的 `video-shot-splitter/src/` 目录正好用上。
- 2036 行的 `video_splitter.py` 拆成 `cli.py` / `pipeline.py` / `report.py` 三块，按职责分离 —— 机械搬迁 + 调整 import，不改算法逻辑。
- `vendor/transnetv2/` 整体搬进包内，继续保留其 LICENSE（合规）。

## 权重打包与资源定位

- 权重放在 `src/video_shot_splitter/models/`，通过 `pyproject.toml` 的 `[tool.setuptools.package-data]` 声明为包数据，确保进 wheel。
- 代码改用 **`importlib.resources`** 定位权重，替换现有的 `Path(__file__).parent / "models" / ...` 硬编码（现 `video_splitter.py:819`）。在 zip 安装、editable 安装、普通安装下都可靠。
- 保留 `--weights` 命令行参数作为覆盖项：默认用包内权重，用户可指定自己的权重。既"装完即用"又保留灵活性。

## 入口点

### 1. CLI
`pyproject.toml` 声明 console script：
```toml
[project.scripts]
video-shot-splitter = "video_shot_splitter.cli:main"
```
装完后终端可用 `video-shot-splitter run video.mov`。同时 `python -m video_shot_splitter` 也可用（靠 `__main__.py`）。

### 2. Python API
`__init__.py` 导出干净的函数：
```python
from video_shot_splitter import run_pipeline
run_pipeline("video.mov", output_dir="out", device="cpu")
```
把 `run_pipeline_cli` 的参数解析与实际逻辑分离，逻辑函数对外暴露，CLI 作为薄包装。

### 3. Web
作为命令 `video-shot-splitter serve` 启动现 `app.py` 的服务。

### 关键修复：subprocess 自调用
现在 `app.py` 在第 216/232/248 行用 `subprocess.Popen([PYTHON, ROOT/"video_splitter.py", stage, ...])` 跑各 stage。打成包后该文件路径不存在。改为：
```python
subprocess.Popen([sys.executable, "-m", "video_shot_splitter", stage, ...])
```
用 `python -m` 调用模块，不依赖文件物理路径，在任何安装方式下都可靠。Web 用 subprocess 跑后台分析、靠日志 tail 显示进度的整体架构**保持不变**。

## 依赖与元数据

pyproject.toml 元数据：
- 包名 `video-shot-splitter`，动态 `version`（从 `__init__.__version__` 读）
- `requires-python = ">=3.10"`
- license、author、description、README 作为 long_description、project URLs 指向 git 仓
- `dependencies`：`numpy`、`opencv-python`、`torch`、`imageio-ffmpeg`、`shotsplit`（沿用 requirements.txt）

**ffmpeg 适配：** `imageio-ffmpeg` 提供 ffmpeg 二进制。让 `ffmpeg_path()`（现 `video_splitter.py:1177`）**优先用 `imageio-ffmpeg` 自带的 ffmpeg**，找不到再退回系统 ffmpeg —— `pip install` 后开箱即用，不用用户另装 ffmpeg。这是打包必要适配，非无关重构。

## 测试与验证

- 保留现有 `smoke-test`（不依赖模型的输出契约自检），重组后接入。
- 验证流程：
  1. `pip install -e .` 装好
  2. `video-shot-splitter smoke-test` 跑通
  3. `python -c "import video_shot_splitter"` 确认 API 可导入
  4. `python -m build` 确认能构建出 wheel 且权重在里面
- **不**新建大型测试框架（YAGNI）；现有 smoke-test 足够验证打包正确性。真实视频端到端测试需模型和样片，留作手动验证。

## 不做的事（YAGNI）

- 不加 CI 发布流水线
- 不加 type stubs
- 不改任何分析算法
- 不动 Web 进度机制
- 不做无关重构
