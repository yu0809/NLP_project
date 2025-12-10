import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import datasets
import evaluate
import numpy as np
import optuna
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
import torch


# ---------------------------
# 数据模型与配置
# ---------------------------
@dataclass
class DataConfig:
    train_file: str
    valid_file: str
    test_file: Optional[str]
    max_source_length: int = 512
    max_target_length: int = 96
    pad_to_max_length: bool = False


def load_jsonl_dataset(train_file: str, valid_file: str, test_file: Optional[str]) -> datasets.DatasetDict:
    data_files = {"train": train_file, "validation": valid_file}
    if test_file:
        data_files["test"] = test_file
    return datasets.load_dataset("json", data_files=data_files)


def load_hf_dataset(dataset_name: str) -> datasets.DatasetDict:
    """从 HuggingFace 加载数据集"""
    raw_datasets = datasets.load_dataset(dataset_name)
    
    # 统一字段名：将 'text' 映射为 'content'（如果存在）
    def rename_text_to_content(example):
        if "text" in example and "content" not in example:
            example["content"] = example["text"]
        return example
    
    # 只对包含 'text' 字段的 split 进行重命名
    for split_name in raw_datasets.keys():
        if "text" in raw_datasets[split_name].column_names and "content" not in raw_datasets[split_name].column_names:
            raw_datasets[split_name] = raw_datasets[split_name].map(rename_text_to_content)
    
    return raw_datasets


def get_preprocess_function(tokenizer, max_source_length: int, max_target_length: int, pad_to_max_length: bool):
    padding = "max_length" if pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples["content"]
        targets = examples["summary"]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            padding=padding,
            truncation=True,
        )
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )
        if padding == "max_length":
            labels["input_ids"] = [
                [(lid if lid != tokenizer.pad_token_id else -100) for lid in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


class Seq2SeqTrainer(Trainer):
    """自定义 Trainer，支持在评估时生成文本"""
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        重写 prediction_step，在评估时使用生成而不是预测 logits
        """
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # 在评估模式下使用生成（当需要预测且没有设置 prediction_loss_only 时）
        if not prediction_loss_only and has_labels:
            with torch.no_grad():
                # 生成摘要
                generated_tokens = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_length=getattr(self.args, "generation_max_length", 96),
                    num_beams=4,
                    early_stopping=True,
                )
                # 计算损失
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else None

            labels = inputs["labels"]
            return (loss, generated_tokens, labels)
        else:
            # 其他情况使用默认行为
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def build_metrics(tokenizer, compute_bertscore: bool, max_target_length: int):
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore") if compute_bertscore else None

    def postprocess_text(texts: List[str]) -> List[str]:
        return [t.strip() for t in texts]

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # 如果 preds 是生成的 token ids（形状为 [batch_size, seq_len]）
        if len(preds.shape) == 2:
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        else:
            decoded_preds = [str(p) for p in preds]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = postprocess_text(decoded_preds)
        decoded_labels = postprocess_text(decoded_labels)

        rouge_result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        result: Dict[str, Any] = {f"rouge_{k}": v for k, v in rouge_result.items()}

        if bertscore:
            bert_res = bertscore.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                lang="zh",
                model_type="bert-base-chinese",
            )
            result["bertscore_f1"] = float(np.mean(bert_res["f1"]))

        if len(preds.shape) == 2:
            result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds])
        return result

    return compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train a summarization model on LCSTS.")
    parser.add_argument("--dataset_name", type=str, default=None, help="HuggingFace dataset name (e.g., 'hugcyp/LCSTS'). If provided, will use this instead of local files.")
    parser.add_argument("--train_file", type=str, default=None, help="Path to train jsonl with fields content/summary (or text/summary)")
    parser.add_argument("--valid_file", type=str, default=None, help="Path to validation jsonl with fields content/summary (or text/summary)")
    parser.add_argument("--test_file", type=str, default=None, help="Optional test jsonl for final eval")
    parser.add_argument("--model_name", type=str, default="fnlp/bart-base-chinese", help="HF model name or local path")
    parser.add_argument("--output_dir", type=str, default="outputs/bart-base-lcsts")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=96)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip_bertscore", action="store_true")
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--use_hpo", action="store_true", help="启用超参数自动优化（使用 Optuna）")
    parser.add_argument("--n_trials", type=int, default=10, help="超参数搜索的试验次数（默认10次）")
    parser.add_argument("--hpo_study_name", type=str, default="summarization_hpo", help="Optuna study 名称")
    return parser.parse_args()


def train_with_params(
    args,
    raw_datasets,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    per_device_train_batch_size: int,
    trial_number: Optional[int] = None,
):
    """使用指定参数进行训练，返回验证集上的 ROUGE-L 分数"""
    data_cfg = DataConfig(
        train_file=args.train_file or "",
        valid_file=args.valid_file or "",
        test_file=args.test_file,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        pad_to_max_length=args.pad_to_max_length,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    preprocess_fn = get_preprocess_function(
        tokenizer=tokenizer,
        max_source_length=data_cfg.max_source_length,
        max_target_length=data_cfg.max_target_length,
        pad_to_max_length=data_cfg.pad_to_max_length,
    )

    column_names = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_fn,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing datasets",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")
    compute_metrics = build_metrics(tokenizer, compute_bertscore=not args.skip_bertscore, max_target_length=args.max_target_length)

    # 为每个 trial 创建独立的输出目录
    output_dir = args.output_dir
    if trial_number is not None:
        output_dir = os.path.join(args.output_dir, f"trial_{trial_number}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=weight_decay,
        save_total_limit=1,  # HPO 时只保留最佳模型
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="rouge_rougeL",
        greater_is_better=True,
        fp16=args.fp16,
        include_inputs_for_metrics=True,
        report_to=None,  # 禁用 wandb/tensorboard
    )

    training_args.generation_max_length = args.max_target_length
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    
    # 返回 ROUGE-L 分数作为优化目标
    rouge_l = eval_metrics.get("eval_rouge_rougeL", 0.0)
    return rouge_l, eval_metrics


def objective(trial, args, raw_datasets):
    """Optuna 优化目标函数"""
    # 定义超参数搜索空间
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16])
    
    print(f"\n🔍 Trial {trial.number}:")
    print(f"   learning_rate: {learning_rate:.2e}")
    print(f"   weight_decay: {weight_decay:.4f}")
    print(f"   warmup_ratio: {warmup_ratio:.4f}")
    print(f"   batch_size: {per_device_train_batch_size}")
    
    try:
        rouge_l, eval_metrics = train_with_params(
            args=args,
            raw_datasets=raw_datasets,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            per_device_train_batch_size=per_device_train_batch_size,
            trial_number=trial.number,
        )
        
        print(f"   ✅ ROUGE-L: {rouge_l:.4f}")
        return rouge_l
    except Exception as e:
        print(f"   ❌ 训练失败: {e}")
        return 0.0  # 返回最低分数，让 Optuna 知道这次试验失败


def main():
    args = parse_args()
    set_seed(args.seed)

    # 加载数据集：优先使用 HuggingFace 数据集
    if args.dataset_name:
        print(f"📥 从 HuggingFace 加载数据集: {args.dataset_name}")
        raw_datasets = load_hf_dataset(args.dataset_name)
        train_columns = raw_datasets["train"].column_names
        if "text" in train_columns:
            print(f"   ✓ 检测到字段: {train_columns}，将自动映射 'text' -> 'content'")
    elif args.train_file and args.valid_file:
        print(f"📁 从本地文件加载数据集")
        raw_datasets = load_jsonl_dataset(args.train_file, args.valid_file, args.test_file)
    else:
        raise ValueError("请提供 --dataset_name 或同时提供 --train_file 和 --valid_file")

    # 超参数优化模式
    if args.use_hpo:
        print(f"\n🚀 启动超参数自动优化（Optuna）")
        print(f"   搜索次数: {args.n_trials}")
        print(f"   优化目标: ROUGE-L (越大越好)\n")
        
        study = optuna.create_study(
            direction="maximize",
            study_name=args.hpo_study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1),
        )
        
        study.optimize(
            lambda trial: objective(trial, args, raw_datasets),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )
        
        print(f"\n✅ 超参数优化完成！")
        print(f"\n📊 最佳参数:")
        print(f"   ROUGE-L: {study.best_value:.4f}")
        print(f"   学习率: {study.best_params['learning_rate']:.2e}")
        print(f"   权重衰减: {study.best_params['weight_decay']:.4f}")
        print(f"   预热比例: {study.best_params['warmup_ratio']:.4f}")
        print(f"   批次大小: {study.best_params['per_device_train_batch_size']}")
        
        # 保存最佳参数
        best_params_file = os.path.join(args.output_dir, "best_params.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(best_params_file, "w", encoding="utf-8") as f:
            json.dump({
                "best_value": study.best_value,
                "best_params": study.best_params,
                "n_trials": args.n_trials,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n💾 最佳参数已保存到: {best_params_file}")
        
        # 使用最佳参数重新训练一次（可选，保存最终模型）
        print(f"\n🔄 使用最佳参数进行最终训练...")
        final_rouge_l, final_metrics = train_with_params(
            args=args,
            raw_datasets=raw_datasets,
            learning_rate=study.best_params['learning_rate'],
            weight_decay=study.best_params['weight_decay'],
            warmup_ratio=study.best_params['warmup_ratio'],
            per_device_train_batch_size=study.best_params['per_device_train_batch_size'],
            trial_number=None,  # 最终训练保存到主目录
        )
        
        print(f"\n✅ 最终模型 ROUGE-L: {final_rouge_l:.4f}")
        print("Training & evaluation completed.")
        return

    # 普通训练模式（原有逻辑）
    data_cfg = DataConfig(
        train_file=args.train_file or "",
        valid_file=args.valid_file or "",
        test_file=args.test_file,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        pad_to_max_length=args.pad_to_max_length,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    preprocess_fn = get_preprocess_function(
        tokenizer=tokenizer,
        max_source_length=data_cfg.max_source_length,
        max_target_length=data_cfg.max_target_length,
        pad_to_max_length=data_cfg.pad_to_max_length,
    )

    column_names = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_fn,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing datasets",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")
    compute_metrics = build_metrics(tokenizer, compute_bertscore=not args.skip_bertscore, max_target_length=args.max_target_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="rouge_rougeL",
        greater_is_better=True,
        fp16=args.fp16,
        include_inputs_for_metrics=True,
    )

    training_args.generation_max_length = args.max_target_length
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final-checkpoint"))
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    if "test" in processed_datasets:
        test_metrics = trainer.evaluate(processed_datasets["test"])
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    print("Training & evaluation completed.")


if __name__ == "__main__":
    main()

