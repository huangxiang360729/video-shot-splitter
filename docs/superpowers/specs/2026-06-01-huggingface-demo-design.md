# 把 web 服务部署为 HuggingFace Spaces 在线 demo — 设计文档

日期：2026-06-01

## 目标

把 `video-shot-splitter` 的本地 web 服务（`src/video_shot_splitter/web/app.py`）部署成一个公开可访问的在线 demo，托管在 HuggingFace Spaces，并从 GitHub 主仓 README 链接过去。用户打开网址即可上传视频、在线查看分镜/转场拆分结果。

## 关键约束

- web 服务依赖后端跑 Python + torch + TransNetV2 模型 + ffmpeg，**GitHub Pages 无法运行**（只能托管静态文件），因此必须用一台真实服务器。
- 选定 **HuggingFace Spaces**：免费 CPU 档（2 vCPU / 16GB）、专为 ML demo 设计、对 torch/模型/ffmpeg 这类重依赖适配好。

## 关键决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 托管平台 | HuggingFace Spaces | 免费、ML 友好、给公开网址 |
| HF SDK | **Docker**（非 Gradio/Streamlit） | 复用现有 http.server 的 `web/app.py`，几乎零重写 |
| 仓库关系 | GitHub 主仓 + HF Space 独立 git 仓 | demo 部署与源码解耦；Space 通过 `pip install video-shot-splitter` 拉 PyPI 包 |
| 上传上限 | **默认 200MB** | 公开 demo 防滥用（本地默认仍 2GB，由环境变量覆盖） |
| 设备 | CPU | HF 免费档无 GPU，PIPELINE 配置本就是 cpu |
| 部署文件位置 | GitHub 主仓 `deploy/huggingface/` | 版本管理 + 方便复制到 HF |

## 整体架构

```text
GitHub 仓库 (源码, 已有)                HuggingFace Space (新建, 跑 demo)
video-shot-splitter         ──链接──>   huggingface.co/spaces/<你>/video-shot-splitter
  README 加 "🤗 Live Demo" 徽章                  │
  deploy/huggingface/ 存部署文件                 Docker 容器:
                                            - pip install video-shot-splitter (PyPI)
                                            - python -m video_shot_splitter serve
                                            - 监听 0.0.0.0:7860
                                          用户上传视频 → 在线看分镜结果
```

两个独立仓库：GitHub 是代码主仓；HF Space 是另一个 git 仓，只放 `Dockerfile` + `README.md`（带 HF 配置头），通过 PyPI 安装包来跑。包发新版后，Space 重新 build 即更新。

## 部署产物

### 1. HF Space 的 `README.md`（带 YAML 配置头）

```yaml
---
title: Video Shot Splitter
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---
```

正文写 demo 说明 + 使用提示（首次冷启动需等待、建议短视频）。

### 2. `Dockerfile`

```dockerfile
FROM python:3.10-slim
# ffmpeg 运行库 + opencv 需要的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir video-shot-splitter
# HF Spaces 要求非 root 用户，且产物目录要可写
RUN useradd -m appuser
USER appuser
WORKDIR /home/appuser
EXPOSE 7860
CMD ["python", "-m", "video_shot_splitter", "serve"]
```

### 3. 主包 `web/app.py` 的 3 处适配（发新版 0.1.1 到 PyPI）

| 适配点 | 现状 | 改为 | 原因 |
|---|---|---|---|
| host | `127.0.0.1`（532 行） | 读环境变量 `VSS_HOST`，默认 `0.0.0.0` | HF 容器外部要能访问 |
| port | 写死 `7860`（532 行） | 读环境变量 `PORT`，默认 `7860` | 平台可能注入端口 |
| 上传上限 | `2GB`（23 行 `MAX_UPLOAD_BYTES`） | 读环境变量 `VSS_MAX_UPLOAD_MB`，默认 `200` | 公开 demo 防滥用 |

三处都用"环境变量覆盖、保留合理默认"，**本地运行行为基本不变**（host 默认从 127.0.0.1 改为 0.0.0.0 是唯一行为变化，对本地无害）。`opencv-python` 的 GUI 库缺失问题由 Dockerfile 装 `libgl1` 解决，不改依赖（YAGNI）。

## 环境变量（在 HF Space 设置里配，不写死）

```
VSS_HOST=0.0.0.0
VSS_MAX_UPLOAD_MB=200
```

端口用默认 7860，与 HF `app_port` 一致。
