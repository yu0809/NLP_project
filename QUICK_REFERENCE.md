# 快速参考指南

## 测试预训练模型（不微调）

```bash
# 测试所有模型
python test_pretrained.py

# 测试单个模型
python test_pretrained.py --model_key 1
python test_pretrained.py --model_key 2
python test_pretrained.py --model_key 3
```

## 微调单个模型

```bash
# 使用默认超参数
python train_with_hp_tuning.py --model_key 1

# 指定超参数
python train_with_hp_tuning.py --model_key 1 --batch_size 8 --learning_rate 3e-5 --num_epochs 3

# 超参数搜索
python train_with_hp_tuning.py --model_key 1 --hp_search
```

## 批量训练所有模型

```bash
# 批量微调（默认超参数）
python batch_train.py

# 批量超参数搜索
python batch_train.py --hp_search
```

## 模型键对应关系

- `1`: google/mt5-base
- `2`: facebook/mbart-large-cc25
- `3`: csebuetnlp/mT5_multilingual_XLSum

## 输出目录结构

```
checkpoints/
├── 1_finetuned/best_model/      # 模型1的最佳权重
├── 2_finetuned/best_model/      # 模型2的最佳权重
├── 3_finetuned/best_model/      # 模型3的最佳权重
├── 1_hp_search/                 # 模型1的超参数搜索结果
│   ├── run_1/
│   ├── run_2/
│   └── search_results.json
└── ...
```

## 超参数说明

### 默认值
- batch_size: 8
- learning_rate: 3e-5
- num_epochs: 3

### 搜索空间（--hp_search）
- batch_size: [4, 8]
- learning_rate: [1e-5, 3e-5, 5e-5]
- num_epochs: [2, 3]

## 自动功能

✅ **自动最优权重保存**: 训练过程中自动保存验证集上ROUGE-L最高的模型  
✅ **自动超参数选择**: 搜索所有组合，自动选择最佳配置  
✅ **自动评估步数**: 根据数据量自动计算评估和保存频率

