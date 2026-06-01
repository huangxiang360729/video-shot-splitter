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
