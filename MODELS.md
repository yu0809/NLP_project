# 模型说明文档

## 支持的预训练模型

本项目支持以下3个预训练模型，用于比较和微调：

### 1. google/mt5-base ⭐推荐
- **大小**: ~2.3GB
- **特点**: 基础模型，平衡性能和速度
- **适用场景**: 
  - 大多数生产环境
  - 需要平衡质量和速度的场景
  - 默认选择
- **性能**: 速度和质量的良好平衡

### 2. facebook/mbart-large-cc25
- **大小**: ~2.5GB
- **特点**: 多语言BART模型，支持中文
- **适用场景**: 
  - 需要多语言支持的场景
  - 尝试不同架构的模型
- **性能**: 在中文摘要任务上表现良好

### 3. csebuetnlp/mT5_multilingual_XLSum
- **大小**: ~2.3GB
- **特点**: 专门用于摘要任务的多语言模型，在XLSum数据集上训练
- **适用场景**: 
  - 专门针对摘要任务优化的场景
  - 需要更好的摘要质量
- **性能**: 在摘要任务上可能表现更好

## 如何选择模型

### 根据需求选择
- **生产环境/默认选择**: `mt5-base`（推荐）
- **尝试不同架构**: `mbart-large-cc25`
- **摘要专用优化**: `mT5_multilingual_XLSum`

### 模型比较建议
建议使用评估脚本对三个模型进行全面比较：
```bash
python evaluate.py --compare --models 1,2,3 --max_samples 100
```

## 模型切换方法

### 方法1: 在主程序中切换
运行 `python main.py`，选择菜单选项 4 "切换模型"

### 方法2: 在代码中指定
```python
from inference import Summarizer

# 使用模型名称
summarizer = Summarizer("google/mt5-base")

# 使用模型键
from config import Config
model_name = Config.AVAILABLE_MODELS["1"]["name"]
summarizer = Summarizer(model_name)
```

### 方法3: 使用模型比较工具
```bash
python compare_models.py --interactive
```

## 模型性能对比

建议使用评估脚本进行实际测试：

```bash
# 比较所有模型
python evaluate.py --compare --max_samples 100

# 比较指定模型
python evaluate.py --compare --models 1,2,3 --max_samples 100
```

评估结果会保存到 `evaluation_results.json` 文件中，包含：
- ROUGE-1 分数
- ROUGE-2 分数
- ROUGE-L 分数
- 评估样本数

## 注意事项

1. **首次下载**: 每个模型首次使用时需要从 Hugging Face 下载，请确保网络连接正常
2. **存储空间**: 确保有足够的磁盘空间存储模型文件
3. **内存要求**: 大型模型需要更多内存，建议至少8GB
4. **GPU加速**: 如果有GPU，所有模型都会自动使用GPU加速
5. **模型缓存**: 下载的模型会缓存在 `~/.cache/huggingface/` 目录

## 添加新模型

如果想添加新的模型，可以在 `config.py` 中的 `AVAILABLE_MODELS` 字典中添加：

```python
AVAILABLE_MODELS = {
    # ... 现有模型 ...
    "4": {
        "name": "your-model-name",
        "description": "模型描述",
        "size": "模型大小"
    }
}
```

确保新模型：
- 支持中文
- 是序列到序列（Seq2Seq）模型
- 兼容 Transformers 库

