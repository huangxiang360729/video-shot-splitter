# Video Shot Splitter

本项目是一个本地视频分镜/转场拆分工具。它会把视频拆成两类 clip：

- `transition`：转场片段
- `normal`：正常镜头片段

支持两种使用方式：

- Web 页面上传分析
- 命令行一键分析

## 目录结构

```text
.
├── app.py              # 本地 Web 上传和进度页面
├── video_splitter.py   # 视频分析主程序
├── requirements.txt    # Python 依赖
├── vendor/transnetv2/  # TransNetV2 PyTorch 推理代码
└── models/             # TransNetV2 权重
```

生成结果默认不会提交到 Git，包括：

```text
outputs*/
web_runs/
*.mov / *.mp4 / ...
```

## 安装环境

建议使用 Python 3.10+。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果你使用 Codex 本地运行时，也可以直接用它的 Python：

```powershell
C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m pip install -r requirements.txt
```

## TransNetV2

仓库已内置 TransNetV2 PyTorch 推理代码：

```text
vendor/transnetv2/transnetv2_pytorch.py
```

仓库也已内置当前使用的 PyTorch 权重文件：

```text
models/
  transnetv2-pytorch-weights.pth
```

`vendor/transnetv2/` 来自 [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2)，保留了原项目的 `LICENSE` 和 PyTorch inference README。

## Web 使用

启动本地网页服务：

```powershell
python app.py
```

打开浏览器：

```text
http://127.0.0.1:7860/
```

页面操作：

1. 上传视频。
2. 点击“开始分析”。
3. 等待页面显示上传和后台分析进度。
4. 完成后自动跳转到结果报告页。

Web 流程会自动使用当前调好的默认参数。

## 命令行一键使用

最简单用法：

```powershell
python video_splitter.py run path\to\video.mov
```

指定输出目录：

```powershell
python video_splitter.py run path\to\video.mov --output-dir outputs_pipeline
```

GPU 可用时可以指定：

```powershell
python video_splitter.py run path\to\video.mov --device cuda
```

## 输出内容

一键命令输出目录默认是：

```text
outputs_pipeline/
├── transnet/
├── autoshot/
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
normal_transition_runs.csv      # 每个 normal/transition run 的表格
run_clips/                      # 最终 normal/transition 小视频
candidate_sweep_report.html     # 候选点调试报告
candidate_sweep.csv             # 候选点分数和来源
candidate_sweep_meta.json       # 候选点元信息
candidate_clips/                # 候选点附近的小视频
```

普通用户主要看：

```text
outputs_pipeline/result/normal_transition_report.html
outputs_pipeline/result/summary.json
```

## summary.json

`summary.json` 是下游程序最适合读取的最终结果文件，结构大致如下：

```json
{
  "video": {
    "path": "path/to/video.mov",
    "fps": 25.0,
    "frame_count": 1000,
    "duration_seconds": 40.0
  },
  "parameters": {
    "candidate_threshold": 0.34
  },
  "outputs": {
    "report_html": "outputs_pipeline/result/normal_transition_report.html",
    "runs_csv": "outputs_pipeline/result/normal_transition_runs.csv",
    "run_clips_dir": "outputs_pipeline/result/run_clips"
  },
  "counts": {
    "total_clips": 12,
    "transition_clips": 5,
    "normal_clips": 7
  },
  "clips": {
    "transition": [],
    "normal": []
  }
}
```

每个 clip 条目包含：

```json
{
  "id": 1,
  "label": "transition",
  "start_frame": 100,
  "end_frame": 110,
  "start_timecode": "00:00:04.000",
  "end_timecode": "00:00:04.400",
  "duration_seconds": 0.4,
  "source": "hard_cut_pad",
  "clip_path": "outputs_pipeline/result/run_clips/run_001_transition_00-00-04.000.mp4"
}
```

## 高级命令

普通使用建议只用 `run`。如果需要单独调试某个阶段，可以使用：

```powershell
python video_splitter.py transnet VIDEO --help
python video_splitter.py autoshot VIDEO --help
python video_splitter.py sweep VIDEO --help
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
