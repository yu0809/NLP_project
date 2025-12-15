# 本地模型使用说明

## 模型缓存机制

### Transformers 自动缓存

Transformers 库会自动将下载的模型缓存到本地，后续使用时会**自动从本地加载**，无需再次下载。

### 缓存位置

默认缓存路径：
- **Linux/Mac**: `~/.cache/huggingface/transformers/`
- **Windows**: `C:\Users\<username>\.cache\huggingface\transformers\`

### 验证模型是否已缓存

运行以下命令查看缓存：
```python
from transformers import file_utils
print(file_utils.default_cache_path)
```

## 下载所有模型

### 方法1: 使用下载脚本（推荐）

```bash
# 下载所有三个模型
python download_model.py

# 下载指定模型
python download_model.py --model_key 1
```

### 方法2: 首次运行时自动下载

如果模型未缓存，首次使用时会自动下载：
- 运行 `test_pretrained.py` 时会自动下载
- 运行 `train_with_hp_tuning.py` 时会自动下载
- 运行 `main.py` 时会自动下载

## 使用本地模型

### 自动使用本地缓存

**所有代码已经配置为优先使用本地模型**：

1. **测试脚本** (`test_pretrained.py`):
   ```python
   # 自动从本地缓存加载
   summarizer = Summarizer(model_name)
   ```

2. **训练脚本** (`train_with_hp_tuning.py`):
   ```python
   # 自动从本地缓存加载
   model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   ```

3. **推理脚本** (`inference.py`):
   ```python
   # 自动从本地缓存加载
   self.tokenizer = AutoTokenizer.from_pretrained(model_path)
   self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
   ```

### Transformers 的加载顺序

`from_pretrained()` 的加载顺序：
1. ✅ **本地缓存**（如果存在）
2. ❌ Hugging Face Hub（仅当本地不存在时）

所以**无需额外配置，代码会自动使用本地模型**。

## 验证本地模型使用

### 方法1: 检查网络活动

运行测试或训练时，如果使用本地模型：
- ✅ 不会显示下载进度条
- ✅ 加载速度很快（几秒内完成）
- ✅ 不会消耗网络流量

### 方法2: 查看缓存目录

```bash
# 查看缓存目录
ls ~/.cache/huggingface/transformers/

# 应该能看到类似这样的目录：
# - google--mt5-base
# - facebook--mbart-large-cc25
# - csebuetnlp--mT5_multilingual_XLSum
```

### 方法3: 离线模式

可以设置环境变量强制使用本地模型：
```bash
export TRANSFORMERS_OFFLINE=1
python test_pretrained.py
```

## 完整工作流程

### 1. 首次使用（需要下载）

```bash
# 方式1: 提前下载所有模型
python download_model.py

# 方式2: 直接使用（会自动下载）
python test_pretrained.py
```

### 2. 后续使用（使用本地缓存）

```bash
# 所有脚本都会自动使用本地缓存
python test_pretrained.py          # 使用本地模型
python train_with_hp_tuning.py --model_key 1  # 使用本地模型
python main.py                     # 使用本地模型
```

## 常见问题

### Q: 如何确认使用的是本地模型？

A: 如果模型已缓存，加载时会：
- 不显示下载进度
- 加载速度很快
- 不消耗网络流量

### Q: 如何强制重新下载？

A: 删除缓存目录或使用 `force_download=True`:
```python
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    force_download=True
)
```

### Q: 如何清理缓存？

A: 删除缓存目录：
```bash
rm -rf ~/.cache/huggingface/transformers/
```

### Q: 缓存占用多少空间？

A: 三个模型大约需要：
- google/mt5-base: ~2.3GB
- facebook/mbart-large-cc25: ~2.5GB
- csebuetnlp/mT5_multilingual_XLSum: ~2.3GB
- **总计**: ~7GB

## 总结

✅ **所有代码已配置为自动使用本地模型**
✅ **首次下载后，后续使用无需网络连接**
✅ **无需额外配置，Transformers 会自动处理**

只需运行 `python download_model.py` 下载所有模型，之后所有操作都会自动使用本地缓存。

