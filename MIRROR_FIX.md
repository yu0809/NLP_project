# 镜像配置问题修复指南

## 问题说明

如果设置了镜像但仍然连接 `huggingface.co`，可能是因为：
1. `HF_ENDPOINT` 环境变量在脚本内部设置，但 transformers 库在导入时已经读取了配置
2. 某些版本的 transformers/huggingface_hub 可能不支持 `HF_ENDPOINT` 环境变量

## 解决方案

### 方法1: 在运行脚本前设置环境变量（推荐）

在运行下载脚本**之前**，先在终端设置环境变量：

```bash
# Linux/Mac
export HF_ENDPOINT=https://hf-mirror.com
python download_model.py

# Windows (PowerShell)
$env:HF_ENDPOINT="https://hf-mirror.com"
python download_model.py
```

### 方法2: 永久设置环境变量

#### Linux/Mac

编辑 `~/.bashrc` 或 `~/.zshrc`：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后执行：
```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

#### Windows

在系统环境变量中添加：
- 变量名: `HF_ENDPOINT`
- 变量值: `https://hf-mirror.com`

### 方法3: 使用脚本包装器

创建一个启动脚本 `download_with_mirror.sh`：

```bash
#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
python download_model.py "$@"
```

然后运行：
```bash
chmod +x download_with_mirror.sh
./download_with_mirror.sh
```

### 方法4: 不使用镜像（如果镜像不工作）

如果镜像配置有问题，可以直接连接：

```bash
python download_model.py --no-mirror
```

## 验证镜像是否生效

运行下载脚本时，检查输出中是否显示：

```
✓ 已设置镜像源: hf-mirror.com (推荐)
  镜像地址: https://hf-mirror.com
  环境变量 HF_ENDPOINT = https://hf-mirror.com
```

如果仍然看到连接 `huggingface.co` 的错误，说明环境变量未生效。

## 快速测试

```bash
# 1. 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 2. 验证设置
echo $HF_ENDPOINT
# 应该输出: https://hf-mirror.com

# 3. 运行下载脚本
python download_model.py --mirror 1
```

## 如果仍然失败

如果所有方法都失败，可以：

1. **检查网络连接**：确保可以访问 hf-mirror.com
2. **使用VPN**：如果镜像不可用，使用VPN连接原始源
3. **手动下载**：从镜像站手动下载模型文件

## 临时解决方案

如果急需使用，可以先不使用镜像：

```bash
python download_model.py --no-mirror
```

虽然速度可能较慢，但可以正常工作。

