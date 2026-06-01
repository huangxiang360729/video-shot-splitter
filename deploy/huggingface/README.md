---
title: Video Shot Splitter
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Video Shot Splitter — 在线 Demo

本地视频分镜/转场拆分工具的在线体验版。上传一段视频，自动把它切成转场片段与正常镜头片段。

## 使用提示

- **首次访问需等待**：Space 闲置后会休眠，第一次打开需要约 30-60 秒唤醒并启动容器。
- **建议用短视频**：本 demo 跑在免费 CPU 实例上，推理较慢，建议上传 30 秒以内的视频体验。
- **上传上限 200MB**。
- 产物在容器重启后不保留，仅供体验。

源码与本地安装：https://github.com/huangxiang360729/video-shot-splitter
（本地运行可处理更大视频、可用 GPU 加速。）
