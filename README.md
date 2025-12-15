# 中文自动摘要系统

基于 Transformer 的生成式自动摘要模型，在 LCSTS 中文数据集上训练。

## 项目简介

本项目实现了一个基于 Transformer 的生成式自动摘要系统，能够对中文文本进行自动摘要生成。系统支持：

- ✅ 生成式摘要（允许生成新词语和短语）
- ✅ 语序连贯、可读性强的摘要输出
- ✅ 任意文本输入摘要生成
- ✅ 非文本输入检测和提示
- ✅ 批量处理功能
- ✅ **多模型支持** - 支持5种不同的预训练模型，可随时切换
- ✅ **模型比较** - 可以同时使用多个模型生成摘要并比较效果
- ✅ **性能评估** - 支持多模型在测试集上的性能评估和比较

## 技术架构

- **模型**: 支持多个预训练模型（mT5系列、mBART等）
- **框架**: PyTorch + Transformers
- **任务类型**: 生成式摘要 (Abstractive Summarization)
- **评价指标**: ROUGE-1, ROUGE-2, ROUGE-L

## 支持的模型

系统支持以下3个预训练模型，用于比较和微调：

1. **google/mt5-base** - 基础模型，平衡性能和速度（推荐）
2. **facebook/mbart-large-cc25** - 多语言BART模型，支持中文
3. **csebuetnlp/mT5_multilingual_XLSum** - 专门用于摘要任务的多语言模型

## 项目结构

```
NLP_project/
├── Pre LCSTS/              # 数据集目录
│   ├── train.csv          # 训练集
│   ├── eval.csv           # 验证集
│   └── test.csv           # 测试集
├── config.py              # 配置文件
├── data_processor.py      # 数据处理模块
├── train.py              # 模型训练脚本
├── inference.py          # 推理模块
├── main.py               # 主程序（交互式界面）
├── compare_models.py     # 模型比较脚本
├── evaluate.py           # 模型评估脚本（支持多模型）
├── requirements.txt      # 依赖包
└── README.md            # 说明文档
```

## 环境要求

- Python >= 3.7
- PyTorch >= 2.0.0
- CUDA (可选，用于GPU加速)

## 安装步骤

1. **克隆或下载项目**

2. **安装依赖包**
```bash
pip install -r requirements.txt
```

3. **下载预训练模型**
模型会在首次运行时自动下载，也可以手动下载：
```bash
# 模型会自动从 Hugging Face 下载
# 模型名称: google/mt5-base
```

## 使用方法

### 1. 训练模型

训练自定义模型（可选，也可以直接使用预训练模型）：

```bash
python train.py
```

训练参数可在 `config.py` 中修改：
- `BATCH_SIZE`: 批次大小
- `LEARNING_RATE`: 学习率
- `NUM_EPOCHS`: 训练轮数
- `MAX_SOURCE_LENGTH`: 原文最大长度
- `MAX_TARGET_LENGTH`: 摘要最大长度

**注意**: 训练集很大（100万+条），建议：
- 使用GPU加速训练
- 首次测试可以限制训练样本数量（代码中已设置 max_samples=50000）

### 2. 使用摘要系统

运行主程序：

```bash
python main.py
```

系统提供三种使用模式：

#### 模式1: 单文本摘要
- 直接输入文本，系统会生成摘要
- 输入 'quit' 退出

#### 模式2: 文件摘要
- 输入文件路径，系统会读取文件内容并生成摘要
- 可选择保存摘要到文件

#### 模式3: 批量处理
- 输入目录路径，系统会批量处理目录中的所有文本文件
- 摘要会保存在 `summaries/` 子目录中

#### 模式4: 切换模型
- 在运行时切换不同的预训练模型
- 支持所有可用的预训练模型

#### 模式5: 模型比较
- 使用多个模型对同一文本生成摘要并比较
- 可以查看不同模型的效果差异

### 3. 模型比较

比较多个模型的摘要生成效果：

```bash
# 交互式比较
python compare_models.py --interactive

# 使用示例文本比较
python compare_models.py "你的文本内容"

# 比较指定模型（模型键用逗号分隔）
python compare_models.py "文本内容" --models 1,2,3
```

### 4. 模型评估

评估模型在测试集上的性能：

```bash
# 评估单个模型
python evaluate.py --model_key 2 --max_samples 100

# 比较多个模型
python evaluate.py --compare --models 1,2,3 --max_samples 100

# 评估所有模型
python evaluate.py --compare --max_samples 100
```

### 5. 编程接口

也可以直接在代码中使用：

```python
from inference import Summarizer
from config import Config

# 使用默认模型
summarizer = Summarizer()

# 使用指定模型
summarizer = Summarizer("google/mt5-base")  # 使用模型名称
summarizer = Summarizer("checkpoints/final_model")  # 使用本地模型路径

# 使用模型键选择
model_name = Config.AVAILABLE_MODELS["2"]["name"]  # 获取模型名称
summarizer = Summarizer(model_name)

# 生成摘要
text = "这是一段需要摘要的长文本..."
summary = summarizer.generate_summary(text)
print(summary)

# 批量生成
texts = ["文本1...", "文本2...", "文本3..."]
summaries = summarizer.batch_generate(texts)
```

## 输入验证

系统会自动检测并拒绝非文本输入：

- ✅ 检查输入是否为空
- ✅ 检查输入类型是否为字符串
- ✅ 检查是否包含有效的中文或英文字符
- ✅ 检查文本长度（至少10个字符）

如果输入无效，系统会提示相应的错误信息。

## 配置说明

主要配置项在 `config.py` 中：

```python
# 模型配置
MODEL_NAME = "google/mt5-base"  # 使用的模型
MAX_SOURCE_LENGTH = 512         # 原文最大长度
MAX_TARGET_LENGTH = 128         # 摘要最大长度

# 推理配置
NUM_BEAMS = 4                   # Beam search的beam数量
LENGTH_PENALTY = 0.6            # 长度惩罚
NO_REPEAT_NGRAM_SIZE = 3        # 避免重复的n-gram大小
```

## 评价指标

训练过程中会计算以下指标：
- **ROUGE-1**: 基于unigram的重叠度
- **ROUGE-2**: 基于bigram的重叠度  
- **ROUGE-L**: 基于最长公共子序列的分数

## 注意事项

1. **首次运行**: 首次运行需要下载预训练模型（约1.2GB），请确保网络连接正常
2. **GPU加速**: 如果有GPU，训练和推理会自动使用GPU加速
3. **内存要求**: 建议至少8GB内存，训练时建议16GB+
4. **训练时间**: 完整训练可能需要数小时（取决于硬件配置）

## 示例输出

```
============================================================
               中文自动摘要系统
============================================================

检测到已训练的模型，将使用: checkpoints/final_model
正在初始化模型，请稍候...
✅ 模型初始化成功！

请选择操作：
1. 输入文本生成摘要
2. 从文件读取文本生成摘要
3. 批量处理文件
4. 退出
------------------------------------------------------------
请选择 (1-4): 1

============================================================
单文本摘要模式
============================================================

请输入要摘要的文本（输入完成后按回车，输入'quit'退出）：
------------------------------------------------------------

输入文本: 人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器...

正在生成摘要...

============================================================
生成的摘要：
============================================================
人工智能是计算机科学分支，旨在了解智能本质并创造类似人类智能的机器。
============================================================
```

## 常见问题

**Q: 模型下载失败怎么办？**
A: 可以手动从 Hugging Face 下载模型，或使用镜像源。

**Q: 训练时内存不足？**
A: 减小 `BATCH_SIZE` 或 `MAX_SOURCE_LENGTH`。

**Q: 摘要质量不理想？**
A: 可以增加训练轮数，或尝试其他模型（如 mBART 或 mT5 XLSum），使用 `python compare_models.py` 比较不同模型的效果。

**Q: 如何评估模型性能？**
A: 训练过程中会自动在验证集上计算ROUGE分数。

## 许可证

本项目仅用于学习和研究目的。

## 参考文献

- LCSTS: A Large Scale Chinese Short Text Summarization Dataset
- mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer
- ROUGE: A Package for Automatic Evaluation of Summaries

