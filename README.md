# 自动摘要课程项目

本项目基于 LCSTS 中文摘要数据集，提供生成式自动摘要的完整训练与推理脚本（使用 HuggingFace Transformers）。默认模型选用 `fnlp/bart-base-chinese`，可按需替换为其他中英文 Seq2Seq 预训练模型。

## 功能概览

- **生成式摘要训练**：支持从 HuggingFace 直接加载 LCSTS 数据集，内置 ROUGE 与可选 BERTScore 评估
- **推理与输入检测**：对任意输入文本生成摘要，并对非文本或过短输入给出友好提示
- **灵活配置**：通过命令行参数快速调整最大长度、批大小、学习率等

## 模型架构

- **模型名称**: `fnlp/bart-base-chinese`
- **模型类型**: BART (Bidirectional and Auto-Regressive Transformers)
- **架构**: Seq2Seq (序列到序列) 生成式模型
- **特点**: 基于 Transformer 的编码器-解码器架构，专门针对中文优化，支持生成新词

## 环境依赖

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 快速开始

### 训练模型（推荐：使用 HuggingFace 数据集）

**普通训练**（使用默认参数）:
```bash
python train.py \
  --dataset_name hugcyp/LCSTS \
  --output_dir outputs/bart-base-lcsts \
  --model_name fnlp/bart-base-chinese \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8
```

**自动超参数优化**（推荐）✨:
```bash
python train.py \
  --dataset_name hugcyp/LCSTS \
  --output_dir outputs/bart-base-lcsts \
  --model_name fnlp/bart-base-chinese \
  --num_train_epochs 3 \
  --use_hpo \
  --n_trials 10
```

**数据集信息**:
- 自动包含 train (2,400,591 条), validation (8,685 条), test (725 条)
- 字段自动映射：`text` → `content`

### 使用本地 JSONL 文件

如果使用本地数据文件：

```bash
python train.py \
  --train_file data/train.jsonl \
  --valid_file data/valid.jsonl \
  --output_dir outputs/bart-base-lcsts \
  --model_name fnlp/bart-base-chinese \
  --num_train_epochs 3
```

数据格式（JSONL，每行一个 JSON 对象）：
```json
{"content": "原文内容", "summary": "对应摘要"}
```

## 训练参数

### 普通训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_train_epochs` | 3 | 训练轮数 |
| `per_device_train_batch_size` | 8 | 训练批次大小 |
| `learning_rate` | 5e-5 | 学习率 |
| `max_source_length` | 512 | 输入文本最大长度（tokens） |
| `max_target_length` | 96 | 输出摘要最大长度（tokens） |
| `eval_steps` | 500 | 每500步评估一次 |
| `logging_steps` | 50 | 每50步记录日志 |

### 超参数自动优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_hpo` | False | 启用超参数自动优化 |
| `--n_trials` | 10 | 搜索试验次数（建议 10-20 次） |

**超参数搜索范围**:
- `learning_rate`: 1e-5 到 1e-4（对数分布）
- `weight_decay`: 0.0 到 0.1
- `warmup_ratio`: 0.0 到 0.2
- `per_device_train_batch_size`: [4, 8, 16]

**优化目标**: ROUGE-L 分数（越大越好）

**其他关键参数**:
- `--skip_bertscore`: 跳过 BERTScore 计算（加快评估速度）
- `--gradient_accumulation_steps`: 梯度累积（显存不足时可设为 2 或 4）
- `--fp16`: 开启半精度训练（GPU 支持时使用）

## 推理使用

### 单条文本摘要

```bash
python predict.py \
  --model_dir outputs/bart-base-lcsts/best-checkpoint \
  --text "输入你要摘要的文本内容"
```

### 批量预测

```bash
python predict.py \
  --model_dir outputs/bart-base-lcsts/best-checkpoint \
  --dataset_file data/test.jsonl \
  --batch_mode \
  --batch_size 8
```

**输入检测**: 自动检测非文本输入（如纯数字、符号等）并提示错误。

## 评估指标

训练过程中自动计算：
- **ROUGE-1**: 基于单词重叠
- **ROUGE-2**: 基于二元组重叠
- **ROUGE-L**: 基于最长公共子序列
- **BERTScore** (可选): 基于语义相似度

模型保存：
- `best-checkpoint/`: 最佳模型（基于 ROUGE-L 分数）
- `final-checkpoint/`: 最终模型（训练结束时的状态）

## 项目结构

```
NLP_project/
├── train.py              # 训练脚本
├── predict.py            # 推理脚本
├── requirements.txt      # 依赖列表
└── README.md            # 本文件
```

## 训练流程

### 普通训练流程

1. **数据加载**: 从 HuggingFace 加载或使用本地 JSONL 文件
2. **数据预处理**: Tokenization，原文截断到 512 tokens，摘要截断到 96 tokens
3. **模型训练**: 使用交叉熵损失，AdamW 优化器
4. **评估**: 每 500 步在验证集上评估，使用 Beam Search (num_beams=4) 生成摘要
5. **模型保存**: 自动保存最佳模型（基于 ROUGE-L 分数）

### 超参数自动优化流程

1. **创建 Optuna Study**: 使用 TPE (Tree-structured Parzen Estimator) 算法
2. **多次试验**: 每次试验使用不同的超参数组合进行训练
3. **评估与选择**: 基于验证集 ROUGE-L 分数选择最佳参数
4. **最终训练**: 使用最佳参数重新训练一次，保存最终模型
5. **保存结果**: 最佳参数保存到 `best_params.json`

**优势**:
- ✅ 自动找到最优超参数组合
- ✅ 无需手动调参
- ✅ 基于实际验证集表现优化

## 推理流程

1. **输入检测**: 检查是否为有效文本（至少4个字符，包含中文或英文）
2. **Tokenization**: 将文本转换为 token IDs
3. **生成摘要**: 使用 Beam Search (num_beams=4) 生成摘要
4. **解码输出**: 将 token IDs 转换回文本

## 常见问题

- **无法联网下载模型/指标**: 请在有网环境预先运行脚本或手动缓存模型、`rouge`、`bertscore` 所需资源到 `~/.cache/huggingface` 后再离线训练
- **显存不足**: 降低 `max_source_length`、`max_target_length`，或使用更小的预训练模型（如 `uer/t5-small-chinese-cluecorpussmall`）
- **数据格式错误**: 确保 jsonl 每行均含 `content` 与 `summary` 字段，且无空字符串

## 技术细节

- **生成方式**: Beam Search (beam_size=4, early_stopping=True)
- **损失函数**: 交叉熵损失（忽略 padding tokens）
- **优化器**: AdamW with 线性预热和衰减
- **评估策略**: 每 500 步评估，自动选择最佳模型
