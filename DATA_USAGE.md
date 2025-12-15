# 数据集使用说明

## 数据集划分和使用

### 数据集文件
- `train.csv` - 训练集（约100万条）
- `eval.csv` - 验证集（约1万条）
- `test.csv` - 测试集（约1千条）

### 训练过程中的数据使用

#### ✅ 训练阶段（train_with_hp_tuning.py）

**使用的数据集**：
1. **训练集 (train.csv)**：
   - 用于模型参数更新
   - 通过反向传播和梯度下降优化模型
   - 每个epoch都会使用全部训练数据

2. **验证集 (eval.csv)**：
   - 用于模型性能评估
   - 在每个评估步骤（eval_steps）计算ROUGE分数
   - 用于选择最佳模型checkpoint
   - **不参与参数更新**

**不使用的数据集**：
- ❌ **测试集 (test.csv)**：训练过程中完全不使用

#### ✅ 测试阶段（test_pretrained.py / evaluate.py）

**使用的数据集**：
- **测试集 (test.csv)**：
  - 仅用于最终性能评估
  - 测试预训练模型或微调后的模型
  - 计算ROUGE和BERTScore指标
  - **不参与任何训练过程**

### 数据流程

```
训练阶段:
  train.csv ──┐
              ├──> 模型训练（参数更新）
  eval.csv ───┘    └──> 模型评估（选择最佳checkpoint）
                            │
                            └──> 保存最佳模型

测试阶段:
  test.csv ───────> 加载模型 ───> 生成摘要 ───> 计算指标
```

### 代码中的实现

#### 训练脚本（train_with_hp_tuning.py）

```python
# 只加载训练集和验证集
train_data = processor.load_data(Config.TRAIN_FILE, max_samples=args.max_train_samples)
valid_data = processor.load_data(Config.VALID_FILE, max_samples=args.max_valid_samples)

# 训练时使用训练集
train_dataset = Dataset.from_dict(train_dataset_dict)

# 评估时使用验证集
valid_dataset = Dataset.from_dict(valid_dataset_dict)

# 测试集完全不使用
# ❌ 没有加载 Config.TEST_FILE
```

#### 测试脚本（test_pretrained.py / evaluate.py）

```python
# 只加载测试集
test_data = processor.load_data(Config.TEST_FILE, max_samples=max_samples)

# 使用测试集评估模型性能
for reference, source in test_data:
    prediction = summarizer.generate_summary(source)
    # 计算ROUGE和BERTScore
```

### 为什么这样划分？

1. **训练集 (train.csv)**：
   - 数据量最大，用于学习数据分布
   - 通过大量样本优化模型参数

2. **验证集 (eval.csv)**：
   - 用于模型选择和超参数调优
   - 监控训练过程，防止过拟合
   - 选择最佳模型checkpoint

3. **测试集 (test.csv)**：
   - 完全独立，用于最终评估
   - 模拟真实应用场景
   - 确保模型泛化能力

### 最佳实践

✅ **正确的做法**：
- 训练时：使用 train.csv 和 eval.csv
- 测试时：使用 test.csv
- 测试集只在最终评估时使用一次

❌ **错误的做法**：
- 在训练过程中使用测试集
- 根据测试集结果调整模型
- 多次在测试集上评估并选择最佳结果

### 总结

**训练过程**：
- ✅ 使用 train.csv 进行训练（参数更新）
- ✅ 使用 eval.csv 进行验证（模型选择）
- ❌ 不使用 test.csv

**测试过程**：
- ✅ 使用 test.csv 进行最终评估
- ✅ 计算ROUGE和BERTScore指标
- ❌ 不参与任何训练

这种划分确保了：
1. 模型在未见过的数据上评估（测试集）
2. 避免数据泄露
3. 获得真实的模型性能指标

