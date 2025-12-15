# 快速启动指南

## 快速开始（3步）

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 直接使用（无需训练）
```bash
python main.py
```
或运行示例：
```bash
python example.py
```

### 3. 训练模型（可选）
如果想在LCSTS数据集上微调模型：
```bash
python train.py
```

## 使用方式

### 方式1: 交互式界面
```bash
python main.py
```
然后选择：
- 1: 直接输入文本生成摘要
- 2: 从文件读取文本生成摘要
- 3: 批量处理目录中的文件

### 方式2: 编程接口
```python
from inference import Summarizer

# 初始化
summarizer = Summarizer()

# 生成摘要
text = "你的文本内容..."
summary = summarizer.generate_summary(text)
print(summary)
```

### 方式3: 评估模型
```bash
python evaluate.py --max_samples 100
```

## 注意事项

1. **首次运行**: 会自动下载预训练模型（约1.2GB），需要网络连接
2. **GPU加速**: 如果有GPU会自动使用，没有GPU也可以运行（速度较慢）
3. **内存要求**: 建议至少8GB内存

## 常见问题

**Q: 模型下载很慢？**
A: 可以使用Hugging Face镜像或手动下载模型

**Q: 训练时内存不足？**
A: 在 `config.py` 中减小 `BATCH_SIZE`

**Q: 想使用训练好的模型？**
A: 训练完成后，模型保存在 `checkpoints/final_model/`，`main.py` 会自动检测并使用

