# Video Shot Splitter

[![🤗 Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/huangxiang360729/video-shot-splitter)

本项目是一个本地视频分镜/转场拆分工具。它会把视频拆成两类 clip：

- `transition`：转场片段
- `normal`：正常镜头片段

支持两种使用方式：

- Web 页面上传分析
- 命令行一键分析

## 处理流程：如何切分与区分 clip

工具把一段视频拆成在时间轴上**首尾相接、完整覆盖**的小片段，每段标为 `transition`（转场）或 `normal`（正常镜头）。整体分三步：

**1. 模型打分（transnet + autoshot）**

对视频逐帧跑两个镜头边界检测模型，各自给出每一帧"这里是不是转场"的分数：

- **TransNetV2**：深度学习镜头边界检测模型。
- **AutoShot**（经 `shotsplit`）：另一套帧级边界检测模型。

两个模型相互独立、互为补充。

**2. 融合与候选点检测（sweep 的前半段）**

把模型分数和本地视觉信号融合，找出"疑似转场"的候选帧：

- **模型分**：取两个模型的较大值 `max(transnet, autoshot)`。
- **视觉分**：由 5 路本地信号加权得到 —— 直方图突变(0.30) + 灰度差(0.20) + 边缘差(0.18) + 闪光/亮度跳变(0.17) + 前后窗口结构差(0.15)。
- **综合分**：取模型分与视觉分的逐帧较大值，在其上找**局部峰值**，并抑制靠得太近的峰（`min-gap`）保证每个转场只留一个候选点。

每个候选点要通过**双重保留规则**才算数（任一满足即保留）：

- 模型分够高（`model keep threshold`）；**或**
- 视觉分够高（`visual keep threshold`）**且**有足够多路信号同时支持（`min-supports`）**且**同时存在结构性变化和瞬时变化。

这套"模型为主、视觉兜底"的双通道设计，既能抓住模型识别的硬切，也能补上模型漏判、但画面明显突变的转场。每个候选点还会按信号特征归类为 `hard_cut`（硬切）、`flash_or_luma`（闪光/亮度跳变）或 `complex_visual`（复杂视觉转场）。

**3. 生成 run 并覆盖时间轴（sweep 的后半段）**

- 每个候选点**向两侧扩展**成一个转场区间（硬切只 pad 几帧，复杂转场按信号扩展，有最大时长上限）。
- 额外把接近**纯黑场/纯白场**的帧并入转场区间。
- 所有转场区间排序、合并重叠 → 这些就是 **`transition` clip**。
- 转场区间**之间的稳定画面** → 自动成为 **`normal` clip**。
- 过短的 normal 间隙会被合并，避免产生碎片。

最终：`transition` 是检测到的边界附近区域，`normal` 是其间的稳定镜头，两者拼起来无缝覆盖整条原视频时间轴。每段都会导出成小视频，并记录进 `summary.json`。

## 目录结构

```text
.
├── pyproject.toml          # 打包配置（PEP 621）
├── src/
│   └── video_shot_splitter/
│       ├── cli.py          # 命令行入口
│       ├── pipeline.py     # 视频分析逻辑 + run_pipeline API
│       ├── report.py       # HTML/可视化报告
│       ├── resources.py    # 定位包内模型权重
│       ├── web/app.py      # 本地 Web 上传和进度页面
│       ├── models/         # TransNetV2 权重（随包分发）
│       └── vendor/transnetv2/  # TransNetV2 PyTorch 推理代码
└── tests/                  # 测试
```

生成结果默认不会提交到 Git，包括：

```text
outputs*/
web_runs/
dist/ build/ *.egg-info/
*.mov / *.mp4 / ...
```

## 安装

需要 Python 3.12（依赖 `shotsplit` 目前仅支持 3.12）。从 PyPI 安装：

```powershell
pip install video-shot-splitter
```

模型权重已随包分发，安装后即可使用，无需额外下载。

从源码开发安装：

```powershell
git clone https://github.com/huangxiang360729/video-shot-splitter
cd video-shot-splitter
pip install -e .
```

## TransNetV2

包内已内置 TransNetV2 PyTorch 推理代码与权重：

```text
src/video_shot_splitter/vendor/transnetv2/transnetv2_pytorch.py
src/video_shot_splitter/models/transnetv2-pytorch-weights.pth
```

权重随 wheel 一起分发，安装后由 `importlib.resources` 自动定位，无需手动指定。需要时仍可用 `--weights` 参数覆盖为自己的权重文件。

`vendor/transnetv2/` 来自 [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2)，保留了原项目的 `LICENSE` 和 PyTorch inference README。

## Web 使用

启动本地网页服务：

```powershell
video-shot-splitter serve
```

打开浏览器：

```text
http://127.0.0.1:7860/
```

上传文件与分析产物会落在你启动服务的当前目录下的 `web_runs/`。

页面操作：

1. 上传视频。
2. 点击“开始分析”。
3. 等待页面显示上传和后台分析进度。
4. 完成后自动跳转到结果报告页。

Web 流程会自动使用当前调好的默认参数。

## 命令行一键使用

最简单用法：

```powershell
video-shot-splitter run path\to\video.mov
```

指定输出目录：

```powershell
video-shot-splitter run path\to\video.mov --output-dir outputs_pipeline
```

GPU 可用时可以指定：

```powershell
video-shot-splitter run path\to\video.mov --device cuda
```

也可以用 `python -m video_shot_splitter run ...` 调用（等价）。

## 输出内容

一键命令输出目录默认是：

```text
outputs_pipeline/
└── result/
```

最重要的结果在：

```text
outputs_pipeline/result/
```

主要文件：

```text
normal_transition_report.html   # 最终可视化报告
summary.json                    # 最终结构化总结
run_clips/                      # 最终 normal/transition 小视频
```

普通用户主要看：

```text
outputs_pipeline/result/normal_transition_report.html
outputs_pipeline/result/summary.json
```

如果使用 `--debug`：

```powershell
video-shot-splitter run path\to\video.mov --output-dir outputs_pipeline --debug
```

会额外保留中间结果和调试输出：

```text
outputs_pipeline/
├── transnet/
├── autoshot/
└── result/
```

`result/` 中会额外包含：

```text
candidate_sweep_report.html     # 候选点调试报告
candidate_sweep.csv             # 候选点分数和来源
candidate_sweep_meta.json       # 候选点元信息
candidate_clips/                # 候选点附近的小视频
normal_transition_runs.csv      # 每个 normal/transition run 的调试表格
```

## summary.json

`summary.json` 是下游程序最适合读取的最终结果文件，结构大致如下：

```json
{
  "schema_version": "1.0",
  "video": {
    "filename": "video.mov",
    "path": "video.mov",
    "fps": 25.0,
    "frame_count": 1000,
    "duration_seconds": 40.0,
    "width": 1920,
    "height": 1080,
    "codec_name": "h264",
    "has_audio": true
  },
  "parameters": {
    "candidate_threshold": 0.34
  },
  "outputs": {
    "report_html": "normal_transition_report.html",
    "run_clips_dir": "run_clips"
  },
  "counts": {
    "total_clips": 12,
    "transition_clips": 5,
    "normal_clips": 7
  },
  "clips": [],
  "clip_groups": {
    "transition": [],
    "normal": []
  }
}
```

每个 clip 条目包含：

```json
{
  "id": 1,
  "type": "transition",
  "start_sec": 4.0,
  "end_sec": 4.4,
  "duration_sec": 0.4,
  "start_frame": 100,
  "end_frame": 110,
  "start_timecode": "00:00:04.000",
  "end_timecode": "00:00:04.400",
  "source": "hard_cut_pad",
  "clip_path": "run_clips/run_001_transition_00-00-04.000.mp4"
}
```

`clips` 是按原视频时间顺序排列的主结果，完整覆盖原视频时间轴；`clip_groups` 只是为了方便按 normal / transition 分组读取。输出路径均相对于 `summary.json` 所在目录，移动整个结果目录后仍然可用。默认情况下不会把本机绝对视频路径写进 `summary.json`。

## 高级命令

普通使用建议只用 `run`。如果需要单独调试某个阶段，可以使用：

```powershell
video-shot-splitter transnet VIDEO --help
video-shot-splitter autoshot VIDEO --help
video-shot-splitter sweep VIDEO --help
```

也可以跑一个不依赖模型的小型输出契约自检：

```powershell
video-shot-splitter smoke-test
```

完整流程内部实际会依次执行：

1. `transnet`：生成 TransNetV2 帧级候选分数。
2. `autoshot`：生成 AutoShot 帧级候选分数。
3. `sweep`：融合模型分数和本地视觉信号，输出 normal/transition clips。

## 当前默认策略

一键 `run` 默认使用当前调好的参数：

- TransNetV2 threshold: `0.1`
- AutoShot threshold: `0.45`
- adaptive sweep threshold: `0.34`
- model keep threshold: `0.36`
- visual keep threshold: `0.66`
- weak visual keep threshold: `0.55`
- short normal gap 会被合并
- 接近黑场/白场的帧会被吸收到 transition
- 输出 normal / transition clips 和 `summary.json`
- 默认不保留候选点调试输出和中间阶段目录；需要排查算法时使用 `--debug`
