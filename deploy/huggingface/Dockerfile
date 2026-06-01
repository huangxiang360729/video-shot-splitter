FROM python:3.10-slim

# ffmpeg 运行库 + opencv 需要的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 安装已发布到 PyPI 的包（含模型权重）
RUN pip install --no-cache-dir "video-shot-splitter>=0.1.1"

# HF Spaces 要求非 root 用户，产物目录要可写
RUN useradd -m appuser
USER appuser
WORKDIR /home/appuser

EXPOSE 7860
CMD ["python", "-m", "video_shot_splitter", "serve"]
