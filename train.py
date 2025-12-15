"""
模型训练脚本
"""
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import Dataset
from data_processor import DataProcessor
from config import Config
import random
import numpy as np


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS 不需要单独的随机种子设置


def compute_metrics(eval_pred, tokenizer):
    """计算评估指标"""
    from rouge_score import rouge_scorer
    
    predictions, labels = eval_pred
    
    # 解码predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # 解码labels（将-100替换为pad_token_id）
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 计算ROUGE分数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(label, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores),
    }


if __name__ == "__main__":
    # 设置随机种子
    set_seed(Config.SEED)
    
    # 创建输出目录
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("=" * 50)
    print("开始训练自动摘要模型")
    print("=" * 50)
    
    # 初始化数据处理器
    print("\n1. 初始化数据处理器...")
    processor = DataProcessor(Config.MODEL_NAME)
    tokenizer = processor.tokenizer
    
    # 加载数据
    print("\n2. 加载训练数据...")
    # 为了快速训练，可以限制样本数，实际使用时可以注释掉max_samples
    train_data = processor.load_data(Config.TRAIN_FILE, max_samples=50000)
    print(f"训练集大小: {len(train_data)}")
    
    print("\n3. 加载验证数据...")
    valid_data = processor.load_data(Config.VALID_FILE, max_samples=5000)
    print(f"验证集大小: {len(valid_data)}")
    
    # 准备数据集
    print("\n4. 准备数据集...")
    train_dataset_dict = processor.prepare_dataset(train_data)
    valid_dataset_dict = processor.prepare_dataset(valid_data)
    
    # 创建Dataset对象
    train_dataset = Dataset.from_dict(train_dataset_dict)
    valid_dataset = Dataset.from_dict(valid_dataset_dict)
    
    # 预处理数据集
    print("\n5. 预处理数据集...")
    train_dataset = train_dataset.map(
        processor.preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="处理训练集"
    )
    
    valid_dataset = valid_dataset.map(
        processor.preprocess_function,
        batched=True,
        remove_columns=valid_dataset.column_names,
        desc="处理验证集"
    )
    
    # 加载模型
    print("\n6. 加载模型...")
    model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        warmup_steps=Config.WARMUP_STEPS,
        logging_steps=Config.LOGGING_STEPS,
        eval_steps=Config.EVAL_STEPS,
        save_steps=Config.SAVE_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # MPS 暂不支持 fp16，只对 CUDA 启用
        report_to="none",
    )
    
    # 创建Trainer
    print("\n7. 创建Trainer...")
    # 创建compute_metrics的包装函数
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, tokenizer)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_wrapper,
    )
    
    # 开始训练
    print("\n8. 开始训练...")
    print(f"模型: {Config.MODEL_NAME}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(valid_dataset)}")
    print(f"批次大小: {Config.BATCH_SIZE}")
    print(f"学习率: {Config.LEARNING_RATE}")
    print(f"训练轮数: {Config.NUM_EPOCHS}")
    print("-" * 50)
    
    trainer.train()
    
    # 保存最终模型
    print("\n9. 保存模型...")
    trainer.save_model(os.path.join(Config.OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(Config.OUTPUT_DIR, "final_model"))
    
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"模型已保存到: {os.path.join(Config.OUTPUT_DIR, 'final_model')}")
    print("=" * 50)

