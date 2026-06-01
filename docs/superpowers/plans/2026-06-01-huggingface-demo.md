# HuggingFace Spaces 在线 demo 部署实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `video-shot-splitter` 的 web 服务部署成 HuggingFace Spaces 公开在线 demo，并从 GitHub README 链接过去。

**Architecture:** 主包 `web/app.py` 加 3 处环境变量适配（host/port/上传上限）后发布 0.1.1 到 PyPI；新建 `deploy/huggingface/`（Dockerfile + HF README）通过 `pip install video-shot-splitter` 拉包跑容器；HF Space 是独立 git 仓，从 GitHub 复制部署文件过去。

**Tech Stack:** Python 3.10、HuggingFace Spaces (Docker SDK)、标准库 http.server、Docker。

---

## 文件结构

```text
video-shot-splitter/
├── src/video_shot_splitter/
│   ├── __init__.py                 # 改：版本 0.1.0 → 0.1.1
│   └── web/app.py                  # 改：3 处环境变量适配 + import os
├── tests/
│   └── test_web_config.py          # 新建：环境变量适配的单元测试
├── deploy/
│   └── huggingface/
│       ├── Dockerfile              # 新建：HF Space 容器描述
│       ├── README.md               # 新建：HF Space 配置头 + 说明
│       └── DEPLOY.md               # 新建：在 HF 建 Space 的手动步骤清单
└── README.md                       # 改：加 "🤗 Live Demo" 徽章
```

**已确认的事实（实现时据此操作）：**
- `app.py` 顶部当前**没有** `import os`，3 处适配需要先加它。
- `MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024` 在第 23 行。
- `main()` 在第 529 行，当前 `host = "127.0.0.1"`、`port = 7860` 写死。
- 当前版本 `__version__ = "0.1.0"`，本次发布升到 `0.1.1`。
- 环境变量名：`VSS_HOST`（默认 `0.0.0.0`）、`PORT`（默认 `7860`）、`VSS_MAX_UPLOAD_MB`（默认 `200`）。
- `twine upload` 与 HF Space 创建是**用户手动执行**的步骤（需用户凭据），计划里给出命令清单但不由 agent 执行。

---

## Task 1: web/app.py 三处环境变量适配（TDD）

**Files:**
- Modify: `src/video_shot_splitter/web/app.py`（加 `import os`、改 `MAX_UPLOAD_BYTES`、改 `main()`）
- Test: `tests/test_web_config.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_web_config.py
import importlib
import os


def _reload_app():
    import video_shot_splitter.web.app as app
    return importlib.reload(app)


def test_max_upload_default(monkeypatch):
    monkeypatch.delenv("VSS_MAX_UPLOAD_MB", raising=False)
    app = _reload_app()
    assert app.MAX_UPLOAD_BYTES == 200 * 1024 * 1024


def test_max_upload_env_override(monkeypatch):
    monkeypatch.setenv("VSS_MAX_UPLOAD_MB", "50")
    app = _reload_app()
    assert app.MAX_UPLOAD_BYTES == 50 * 1024 * 1024


def test_server_config_defaults(monkeypatch):
    monkeypatch.delenv("VSS_HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    app = _reload_app()
    host, port = app.server_config()
    assert host == "0.0.0.0"
    assert port == 7860


def test_server_config_env_override(monkeypatch):
    monkeypatch.setenv("VSS_HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "8000")
    app = _reload_app()
    host, port = app.server_config()
    assert host == "127.0.0.1"
    assert port == 8000
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_web_config.py -v`
Expected: FAIL（`server_config` 不存在；`MAX_UPLOAD_BYTES` 仍是 2GB）。

- [ ] **Step 3: 加 `import os`**

在 `app.py` 顶部 import 区（`import mimetypes` 之后那一组里）加一行：

```python
import os
```

- [ ] **Step 4: 改 `MAX_UPLOAD_BYTES`（第 23 行）**

把：
```python
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
```
改为：
```python
MAX_UPLOAD_BYTES = int(os.environ.get("VSS_MAX_UPLOAD_MB", "200")) * 1024 * 1024
```

- [ ] **Step 5: 加 `server_config()` 并改 `main()`**

把 `main()`（第 529 行起）整体改为：

```python
def server_config():
    host = os.environ.get("VSS_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    return host, port


def main():
    RUNS_DIR.mkdir(exist_ok=True)
    host, port = server_config()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Open http://{host}:{port}")
    server.serve_forever()
```

- [ ] **Step 6: 运行确认通过**

Run: `pytest tests/test_web_config.py -v`
Expected: 4 个测试全部 PASS。

- [ ] **Step 7: 跑全量测试确保没破坏其它**

Run: `pytest tests/ -v`
Expected: 全绿（test_resources、test_api、test_web_config）。

- [ ] **Step 8: 提交**

```bash
git add src/video_shot_splitter/web/app.py tests/test_web_config.py
git commit -m "feat: configurable host/port/upload-limit via env vars for deployment"
```

---

## Task 2: 升版本号到 0.1.1

**Files:**
- Modify: `src/video_shot_splitter/__init__.py`

- [ ] **Step 1: 改版本号**

把 `src/video_shot_splitter/__init__.py` 里：
```python
__version__ = "0.1.0"
```
改为：
```python
__version__ = "0.1.1"
```

- [ ] **Step 2: 确认导入版本正确**

Run: `python -c "import video_shot_splitter; print(video_shot_splitter.__version__)"`
Expected: 打印 `0.1.1`（若是 editable 安装无需重装；否则先 `pip install -e .`）。

- [ ] **Step 3: 提交**

```bash
git add src/video_shot_splitter/__init__.py
git commit -m "chore: bump version to 0.1.1"
```

---

## Task 3: 创建 deploy/huggingface 部署文件

**Files:**
- Create: `deploy/huggingface/Dockerfile`
- Create: `deploy/huggingface/README.md`
- Create: `deploy/huggingface/DEPLOY.md`

- [ ] **Step 1: 写 Dockerfile**

`deploy/huggingface/Dockerfile` 内容正好是：

```dockerfile
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
```

- [ ] **Step 2: 写 HF Space README.md（带 YAML 配置头）**

`deploy/huggingface/README.md` 内容正好是：

```markdown
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
```

- [ ] **Step 3: 写 DEPLOY.md（手动部署步骤清单，给用户照做）**

`deploy/huggingface/DEPLOY.md` 内容正好是：

```markdown
# 部署到 HuggingFace Spaces（手动步骤）

> 前置：`video-shot-splitter` 的目标版本已发布到 PyPI（见仓库根的发布流程）。

## 1. 创建 Space

1. 登录 https://huggingface.co （没有账号先注册）
2. 右上头像 → New Space
3. 填写：
   - Owner：你的用户名
   - Space name：`video-shot-splitter`
   - License：mit
   - **SDK：选 Docker → Blank**
   - Hardware：CPU basic（免费）
   - 公开/私有：Public
4. Create Space

## 2. 推送部署文件

HF 会给一个 git 仓地址，形如 `https://huggingface.co/spaces/<你>/video-shot-splitter`。

```bash
git clone https://huggingface.co/spaces/<你>/video-shot-splitter hf-space
cd hf-space
# 从本仓库 deploy/huggingface/ 复制这两个文件进来
cp /path/to/video-shot-splitter/deploy/huggingface/Dockerfile .
cp /path/to/video-shot-splitter/deploy/huggingface/README.md .
git add Dockerfile README.md
git commit -m "Deploy video-shot-splitter demo"
git push
```

> 推送时 HF 要求登录：用户名填你的 HF 用户名，密码填 HF **Access Token**
> （在 https://huggingface.co/settings/tokens 生成，选 write 权限）。

## 3. 配置环境变量（可选，已有合理默认）

Space 页面 → Settings → Variables and secrets，按需添加：

- `VSS_MAX_UPLOAD_MB` = `200`（调整上传上限）

host 默认 `0.0.0.0`、port 默认 `7860` 已适配 HF，无需设置。

## 4. 等待 build 并访问

推送后 HF 自动 build 镜像（含 torch，约几分钟）。build 成功后页面顶部出现 demo，
公开地址即 `https://huggingface.co/spaces/<你>/video-shot-splitter`。
```

- [ ] **Step 4: 本地校验 Dockerfile 语法（若本机有 Docker）**

Run: `docker build -t vss-demo-test deploy/huggingface/`
Expected: build 成功。若本机无 Docker，跳过本步，在实际 HF build 时验证。

- [ ] **Step 5: 提交**

```bash
git add deploy/huggingface/
git commit -m "feat: add HuggingFace Spaces deployment files"
```

---

## Task 4: GitHub README 加 Live Demo 徽章

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 在 README 顶部标题下加徽章**

在 `README.md` 第一个标题（`# ...`）的紧下一行，插入一行（把 `<你>` 换成实际 HF 用户名；若 Space 尚未建好，本步可在部署后回填，但先放占位链接）：

```markdown
[![🤗 Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/huangxiang360729/video-shot-splitter)
```

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "docs: add HuggingFace live demo badge"
```

---

## Task 5: 发布 0.1.1 到 PyPI（用户手动执行）

**Files:** 无代码改动，纯发布。

> 这一步需要用户的 PyPI token，由**用户本人**执行；agent 只负责构建产物和校验。

- [ ] **Step 1: 清理旧产物并构建**

```bash
rm -rf dist build src/*.egg-info
python -m build
```
Expected: 生成 `dist/video_shot_splitter-0.1.1-py3-none-any.whl` 和 `.tar.gz`。

- [ ] **Step 2: 校验产物**

Run: `python -m twine check dist/*`
Expected: 两个文件都 PASSED。

确认权重在内：
```bash
python -c "import glob,zipfile; w=glob.glob('dist/*.whl')[0]; z=zipfile.ZipFile(w); print([n for n in z.namelist() if n.endswith('.pth')])"
```
Expected: 输出含 `video_shot_splitter/models/transnetv2-pytorch-weights.pth`。

- [ ] **Step 3: 用户上传到 PyPI**

```bash
python -m twine upload dist/*
```
（username 填 `__token__`，password 填 PyPI token。版本 0.1.1 不可与已有版本重复。）

- [ ] **Step 4: 验证可安装**

等 PyPI 同步后：
```bash
pip install --upgrade video-shot-splitter
python -c "import video_shot_splitter; print(video_shot_splitter.__version__)"
```
Expected: `0.1.1`。

---

## 验收标准

- `pytest tests/` 全绿（含新的 test_web_config）。
- 设 `VSS_HOST`/`PORT`/`VSS_MAX_UPLOAD_MB` 能覆盖默认；不设时 host=0.0.0.0、port=7860、上限=200MB。
- `deploy/huggingface/` 三文件齐全，Dockerfile 能 build。
- 0.1.1 发布到 PyPI 且可 `pip install`。
- 在 HF 建好 Space 后，公开网址能打开并完成一次短视频分析。
- GitHub README 有可点击的 Live Demo 徽章。
