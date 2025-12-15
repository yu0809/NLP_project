# 国内镜像配置指南

## 快速使用

### 方法1: 使用下载脚本（推荐）

下载脚本会自动配置镜像：

```bash
# 使用国内镜像下载所有模型（默认）
python download_model.py

# 不使用镜像
python download_model.py --no-mirror

# 下载指定模型
python download_model.py --model_key 1
```

### 方法2: 手动设置环境变量

在运行任何脚本之前设置：

```bash
# Linux/Mac
export HF_ENDPOINT=https://hf-mirror.com

# Windows (PowerShell)
$env:HF_ENDPOINT="https://hf-mirror.com"

# Windows (CMD)
set HF_ENDPOINT=https://hf-mirror.com
```

设置后，所有脚本（test_pretrained.py、train_with_hp_tuning.py等）都会自动使用镜像。

## 可用的镜像源

### 1. hf-mirror.com（推荐）

- **地址**: https://hf-mirror.com
- **特点**: 国内访问速度快，稳定可靠
- **使用方法**: 
  - 运行 `python download_model.py` 时选择选项1
  - 或设置环境变量 `HF_ENDPOINT=https://hf-mirror.com`

## 使用方法

### 下载脚本

```bash
# 交互式选择镜像
python download_model.py
# 会提示选择镜像源

# 直接使用镜像（默认）
python download_model.py --mirror 1

# 不使用镜像
python download_model.py --no-mirror
```

### 其他脚本

设置环境变量后，所有脚本都会自动使用镜像：

```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 之后所有脚本都会使用镜像
python test_pretrained.py
python train_with_hp_tuning.py --model_key 1
python main.py
```

## 永久配置（可选）

### Linux/Mac

在 `~/.bashrc` 或 `~/.zshrc` 中添加：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后执行：
```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

### Windows

在系统环境变量中添加：
- 变量名: `HF_ENDPOINT`
- 变量值: `https://hf-mirror.com`

## 验证镜像是否生效

运行下载脚本时，会显示：

```
✓ 已设置镜像源: hf-mirror.com (推荐)
  镜像地址: https://hf-mirror.com
```

或者在Python中检查：

```python
import os
print(os.environ.get('HF_ENDPOINT', '未设置'))
# 应该输出: https://hf-mirror.com
```

## 常见问题

### Q: 镜像下载失败怎么办？

A: 可以尝试：
1. 切换到不使用镜像：`python download_model.py --no-mirror`
2. 检查网络连接
3. 稍后重试（镜像服务器可能暂时繁忙）

### Q: 如何清除镜像设置？

A: 
```bash
# Linux/Mac
unset HF_ENDPOINT

# Windows (PowerShell)
Remove-Item Env:HF_ENDPOINT
```

### Q: 镜像会影响已下载的模型吗？

A: 不会。镜像只影响下载过程，已缓存的模型不受影响。

### Q: 其他脚本也会使用镜像吗？

A: 是的。设置 `HF_ENDPOINT` 环境变量后，所有使用 `transformers` 库的代码都会自动使用镜像。

## 注意事项

1. **镜像只影响下载**：模型下载后会自动缓存到本地，后续使用不依赖镜像
2. **首次下载需要网络**：即使使用镜像，首次下载仍需要网络连接
3. **离线使用**：模型下载完成后，可以离线使用所有功能

## 总结

✅ **推荐做法**：
```bash
# 使用镜像下载所有模型
python download_model.py
# 选择选项1使用hf-mirror.com镜像
```

✅ **后续使用**：
- 模型已缓存到本地
- 所有脚本自动使用本地模型
- 无需再次配置镜像

