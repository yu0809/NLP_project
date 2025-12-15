"""
带超参数调优的训练脚本
支持自动超参数搜索和最优模型权重保存
"""
import os
import torch
import json
from datetime import datetime
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
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
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


def train_model(
    model_name: str,
    model_key: str,
    output_dir: str,
    train_data,
    valid_data,
    hyperparams: dict = None
):
    """
    训练单个模型
    
    Args:
        model_name: 模型名称
        model_key: 模型键
        output_dir: 输出目录
        train_data: 训练数据
        valid_data: 验证数据
        hyperparams: 超参数字典
    """
    print(f"\n{'=' * 80}")
    print(f"训练模型: {model_name}")
    print(f"{'=' * 80}")
    
    # 使用超参数或默认值
    batch_size = hyperparams.get('batch_size', Config.BATCH_SIZE) if hyperparams else Config.BATCH_SIZE
    learning_rate = hyperparams.get('learning_rate', Config.LEARNING_RATE) if hyperparams else Config.LEARNING_RATE
    num_epochs = hyperparams.get('num_epochs', Config.NUM_EPOCHS) if hyperparams else Config.NUM_EPOCHS
    warmup_steps = hyperparams.get('warmup_steps', Config.WARMUP_STEPS) if hyperparams else Config.WARMUP_STEPS
    
    print(f"\n超参数设置:")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  预热步数: {warmup_steps}")
    
    # 初始化数据处理器
    processor = DataProcessor(model_name)
    tokenizer = processor.tokenizer
    
    # 准备数据集
    train_dataset_dict = processor.prepare_dataset(train_data)
    valid_dataset_dict = processor.prepare_dataset(valid_data)
    
    train_dataset = Dataset.from_dict(train_dataset_dict)
    valid_dataset = Dataset.from_dict(valid_dataset_dict)
    
    # 预处理数据集
    print("\n预处理数据集...")
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
    print("\n加载模型...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # 计算评估和保存步数
    total_steps = len(train_dataset) // batch_size * num_epochs
    eval_steps = max(100, total_steps // 20)  # 每个epoch评估约5次
    save_steps = eval_steps
    logging_steps = max(10, total_steps // 100)
    
    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,  # 自动加载最佳模型
        metric_for_best_model="rougeL",  # 使用ROUGE-L作为最佳模型指标
        greater_is_better=True,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=3,  # 只保留最近3个checkpoint
    )
    
    # 创建compute_metrics的包装函数
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, tokenizer)
    
    # 创建Trainer
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
    print("\n开始训练...")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(valid_dataset)}")
    print(f"总步数: {total_steps}")
    print("-" * 80)
    
    train_result = trainer.train()
    
    # 获取最佳模型指标
    best_metrics = trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else None
    best_model_checkpoint = trainer.state.best_model_checkpoint if hasattr(trainer.state, 'best_model_checkpoint') else None
    
    # 保存最终模型（最佳模型已在训练过程中自动保存）
    final_model_dir = os.path.join(output_dir, "best_model")
    print(f"\n保存最佳模型到: {final_model_dir}")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # 保存训练信息
    training_info = {
        'model_name': model_name,
        'model_key': model_key,
        'hyperparams': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'warmup_steps': warmup_steps,
        },
        'best_rougeL': best_metrics,
        'best_checkpoint': best_model_checkpoint,
        'train_loss': train_result.training_loss if hasattr(train_result, 'training_loss') else None,
        'timestamp': datetime.now().isoformat()
    }
    
    info_file = os.path.join(final_model_dir, "training_info.json")
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练完成！")
    print(f"最佳ROUGE-L: {best_metrics:.4f}" if best_metrics else "最佳ROUGE-L: N/A")
    print(f"模型已保存到: {final_model_dir}")
    print(f"训练信息已保存到: {info_file}")
    
    return training_info


def hyperparameter_search(
    model_name: str,
    model_key: str,
    train_data,
    valid_data,
    search_space: dict = None
):
    """
    超参数搜索
    
    Args:
        model_name: 模型名称
        model_key: 模型键
        train_data: 训练数据
        valid_data: 验证数据
        search_space: 超参数搜索空间
    """
    if search_space is None:
        # 默认搜索空间
        search_space = {
            'batch_size': [4, 8, 16],
            'learning_rate': [1e-5, 3e-5, 5e-5],
            'num_epochs': [2, 3],  # 为了节省时间，只搜索少量epoch
        }
    
    print(f"\n{'=' * 80}")
    print(f"超参数搜索: {model_name}")
    print(f"{'=' * 80}")
    print(f"\n搜索空间:")
    for key, values in search_space.items():
        print(f"  {key}: {values}")
    
    best_score = -1
    best_hyperparams = None
    best_training_info = None
    results = []
    
    # 网格搜索
    import itertools
    keys = search_space.keys()
    values = search_space.values()
    
    total_combinations = len(list(itertools.product(*values)))
    print(f"\n总共 {total_combinations} 种超参数组合")
    
    for i, combination in enumerate(itertools.product(*values), 1):
        hyperparams = dict(zip(keys, combination))
        
        print(f"\n{'=' * 80}")
        print(f"组合 {i}/{total_combinations}: {hyperparams}")
        print(f"{'=' * 80}")
        
        output_dir = os.path.join(
            Config.OUTPUT_DIR,
            f"{model_key}_hp_search",
            f"run_{i}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            training_info = train_model(
                model_name=model_name,
                model_key=model_key,
                output_dir=output_dir,
                train_data=train_data,
                valid_data=valid_data,
                hyperparams=hyperparams
            )
            
            score = training_info.get('best_rougeL', 0)
            results.append({
                'hyperparams': hyperparams,
                'score': score,
                'training_info': training_info
            })
            
            if score > best_score:
                best_score = score
                best_hyperparams = hyperparams
                best_training_info = training_info
            
            print(f"\n当前最佳ROUGE-L: {best_score:.4f} (组合 {i})")
        
        except Exception as e:
            print(f"❌ 训练失败: {str(e)}")
            continue
    
    # 保存搜索结果
    search_results = {
        'model_name': model_name,
        'model_key': model_key,
        'search_space': search_space,
        'best_hyperparams': best_hyperparams,
        'best_score': best_score,
        'best_training_info': best_training_info,
        'all_results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = os.path.join(
        Config.OUTPUT_DIR,
        f"{model_key}_hp_search",
        "search_results.json"
    )
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(search_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 80}")
    print("超参数搜索完成")
    print(f"{'=' * 80}")
    print(f"\n最佳超参数: {best_hyperparams}")
    print(f"最佳ROUGE-L: {best_score:.4f}")
    print(f"搜索结果已保存到: {results_file}")
    
    return search_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练模型（支持超参数调优）')
    parser.add_argument('--model_key', type=str, required=True, help='模型键（1-3）')
    parser.add_argument('--hp_search', action='store_true', help='是否进行超参数搜索')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--max_train_samples', type=int, default=None, help='最大训练样本数')
    parser.add_argument('--max_valid_samples', type=int, default=None, help='最大验证样本数')
    
    args = parser.parse_args()
    
    if args.model_key not in Config.AVAILABLE_MODELS:
        print(f"❌ 无效的模型键: {args.model_key}")
        print(f"可用模型键: {list(Config.AVAILABLE_MODELS.keys())}")
        exit(1)
    
    # 设置随机种子
    set_seed(Config.SEED)
    
    # 获取模型信息
    model_info = Config.AVAILABLE_MODELS[args.model_key]
    model_name = model_info["name"]
    
    # 创建输出目录
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    processor = DataProcessor(model_name)
    train_data = processor.load_data(Config.TRAIN_FILE, max_samples=args.max_train_samples)
    valid_data = processor.load_data(Config.VALID_FILE, max_samples=args.max_valid_samples)
    
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(valid_data)}")
    
    # 构建超参数字典
    hyperparams = {}
    if args.batch_size:
        hyperparams['batch_size'] = args.batch_size
    if args.learning_rate:
        hyperparams['learning_rate'] = args.learning_rate
    if args.num_epochs:
        hyperparams['num_epochs'] = args.num_epochs
    
    if args.hp_search:
        # 超参数搜索
        search_space = {
            'batch_size': [4, 8] if not args.batch_size else [args.batch_size],
            'learning_rate': [1e-5, 3e-5, 5e-5] if not args.learning_rate else [args.learning_rate],
            'num_epochs': [2, 3] if not args.num_epochs else [args.num_epochs],
        }
        hyperparameter_search(model_name, args.model_key, train_data, valid_data, search_space)
    else:
        # 单次训练
        output_dir = os.path.join(Config.OUTPUT_DIR, f"{args.model_key}_finetuned")
        train_model(
            model_name=model_name,
            model_key=args.model_key,
            output_dir=output_dir,
            train_data=train_data,
            valid_data=valid_data,
            hyperparams=hyperparams if hyperparams else None
        )

