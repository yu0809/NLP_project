# 训练和测试指南

本指南说明如何使用三个预训练模型进行测试和微调。

## 一、测试预训练模型（不微调）

### 1. 测试单个模型

```bash
# 测试模型1 (mt5-base)
python test_pretrained.py --model_key 1 --max_samples 100

# 测试模型2 (mbart-large-cc25)
python test_pretrained.py --model_key 2 --max_samples 100

# 测试模型3 (mT5_multilingual_XLSum)
python test_pretrained.py --model_key 3 --max_samples 100
```

### 2. 测试所有模型

```bash
# 测试所有三个预训练模型
python test_pretrained.py --max_samples 100

# 使用完整测试集（不限制样本数）
python test_pretrained.py
```

**输出**:
- 控制台显示每个模型的ROUGE分数
- 结果保存到 `pretrained_test_results.json`

## 二、微调单个模型

### 1. 使用默认超参数微调

```bash
# 微调模型1
python train_with_hp_tuning.py --model_key 1

# 微调模型2
python train_with_hp_tuning.py --model_key 2

# 微调模型3
python train_with_hp_tuning.py --model_key 3
```

### 2. 指定超参数微调

```bash
# 自定义批次大小、学习率、训练轮数
python train_with_hp_tuning.py --model_key 1 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --num_epochs 3

# 限制训练样本数（用于快速测试）
python train_with_hp_tuning.py --model_key 1 \
    --max_train_samples 10000 \
    --max_valid_samples 1000
```

### 3. 超参数自动搜索

```bash
# 对模型1进行超参数搜索
python train_with_hp_tuning.py --model_key 1 --hp_search

# 超参数搜索会尝试以下组合：
# - batch_size: [4, 8]
# - learning_rate: [1e-5, 3e-5, 5e-5]
# - num_epochs: [2, 3]
# 总共 2 × 3 × 2 = 12 种组合
```

**输出目录结构**:
```
checkpoints/
├── 1_finetuned/          # 模型1的微调结果
│   └── best_model/       # 最佳模型权重（自动保存）
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer files...
│       └── training_info.json  # 训练信息
└── 1_hp_search/          # 超参数搜索结果
    ├── run_1/            # 第1种超参数组合
    ├── run_2/            # 第2种超参数组合
    ├── ...
    └── search_results.json  # 搜索结果汇总
```

## 三、批量训练所有模型

### 1. 批量微调（使用默认超参数）

```bash
python batch_train.py
```

### 2. 批量微调（指定超参数）

```bash
python batch_train.py \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --num_epochs 3
```

### 3. 批量超参数搜索

```bash
# 对所有三个模型进行超参数搜索
python batch_train.py --hp_search
```

**注意**: 批量超参数搜索会花费很长时间，建议：
- 先限制训练样本数：`--max_train_samples 10000`
- 或者逐个模型进行搜索

## 四、超参数设置说明

### 默认超参数（在 `config.py` 中）

```python
BATCH_SIZE = 8          # 批次大小
LEARNING_RATE = 3e-5    # 学习率
NUM_EPOCHS = 3          # 训练轮数
WARMUP_STEPS = 1000     # 预热步数
```

### 超参数搜索空间（默认）

```python
{
    'batch_size': [4, 8],           # 2种选择
    'learning_rate': [1e-5, 3e-5, 5e-5],  # 3种选择
    'num_epochs': [2, 3]            # 2种选择
}
# 总共 2 × 3 × 2 = 12 种组合
```

### 推荐超参数

根据模型大小和数据集大小调整：

**小数据集（< 10K样本）**:
- batch_size: 4-8
- learning_rate: 5e-5
- num_epochs: 5-10

**中等数据集（10K-100K样本）**:
- batch_size: 8-16
- learning_rate: 3e-5
- num_epochs: 3-5

**大数据集（> 100K样本）**:
- batch_size: 16-32
- learning_rate: 1e-5 to 3e-5
- num_epochs: 2-3

## 五、自动功能说明

### 1. 自动最优权重保存

训练脚本会自动：
- ✅ 在验证集上评估模型性能
- ✅ 使用ROUGE-L作为最佳模型指标
- ✅ 自动保存验证集上表现最好的模型权重
- ✅ 训练结束后自动加载最佳模型

**保存位置**: `checkpoints/{model_key}_finetuned/best_model/`

### 2. 自动超参数选择

使用 `--hp_search` 参数时：
- ✅ 自动尝试所有超参数组合
- ✅ 记录每种组合的验证集性能
- ✅ 自动选择ROUGE-L最高的组合
- ✅ 保存最佳超参数和对应的模型

**保存位置**: `checkpoints/{model_key}_hp_search/search_results.json`

### 3. 自动评估步数计算

训练脚本会自动计算：
- `eval_steps`: 每个epoch评估约5次
- `save_steps`: 与eval_steps相同
- `logging_steps`: 每个epoch记录约100次

## 六、测试微调后的模型

### 1. 测试单个微调模型

```bash
# 测试模型1的微调结果
python evaluate.py --model_path checkpoints/1_finetuned/best_model

# 测试模型2的微调结果
python evaluate.py --model_path checkpoints/2_finetuned/best_model
```

### 2. 比较所有微调模型

```bash
# 先确保所有模型都已微调完成
# 然后使用评估脚本比较
python evaluate.py --compare \
    --models checkpoints/1_finetuned/best_model,checkpoints/2_finetuned/best_model,checkpoints/3_finetuned/best_model
```

## 七、完整工作流程示例

### 场景1: 快速测试预训练模型

```bash
# 1. 测试所有预训练模型（不微调）
python test_pretrained.py --max_samples 100

# 2. 查看结果
cat pretrained_test_results.json
```

### 场景2: 微调并比较

```bash
# 1. 微调模型1（使用默认超参数）
python train_with_hp_tuning.py --model_key 1 --max_train_samples 10000

# 2. 微调模型2
python train_with_hp_tuning.py --model_key 2 --max_train_samples 10000

# 3. 微调模型3
python train_with_hp_tuning.py --model_key 3 --max_train_samples 10000

# 4. 测试微调后的模型
python evaluate.py --model_path checkpoints/1_finetuned/best_model --max_samples 100
python evaluate.py --model_path checkpoints/2_finetuned/best_model --max_samples 100
python evaluate.py --model_path checkpoints/3_finetuned/best_model --max_samples 100
```

### 场景3: 超参数搜索找到最佳配置

```bash
# 1. 对模型1进行超参数搜索（限制样本数以节省时间）
python train_with_hp_tuning.py --model_key 1 --hp_search --max_train_samples 10000

# 2. 查看搜索结果
cat checkpoints/1_hp_search/search_results.json

# 3. 使用最佳超参数进行完整训练
python train_with_hp_tuning.py --model_key 1 \
    --batch_size <最佳batch_size> \
    --learning_rate <最佳learning_rate> \
    --num_epochs <最佳num_epochs>
```

## 八、注意事项

1. **内存要求**: 
   - 每个模型约需2-3GB显存（GPU）或内存（CPU）
   - 批量训练时注意内存限制

2. **训练时间**:
   - 单个模型微调（10K样本）: 约30分钟-1小时（GPU）
   - 超参数搜索（12种组合）: 约6-12小时（GPU）

3. **数据限制**:
   - 默认使用全部数据，可通过 `--max_train_samples` 限制
   - 建议先用小样本测试，确认流程无误后再用全量数据

4. **模型保存**:
   - 最佳模型自动保存在 `best_model/` 目录
   - 训练信息保存在 `training_info.json`
   - 超参数搜索结果保存在 `search_results.json`

5. **中断恢复**:
   - 如果训练中断，可以从checkpoint恢复（需要修改代码）
   - 建议使用 `screen` 或 `tmux` 运行长时间训练任务

## 九、结果文件说明

### pretrained_test_results.json
预训练模型测试结果，包含：
- 每个模型的ROUGE分数
- 测试样本数
- 时间戳

### training_info.json
单次训练信息，包含：
- 使用的超参数
- 最佳ROUGE-L分数
- 训练损失
- 最佳checkpoint路径

### search_results.json
超参数搜索结果，包含：
- 搜索空间
- 最佳超参数组合
- 最佳分数
- 所有组合的结果

