import argparse
import json
import os
import re
from typing import Iterable, List, Tuple

import datasets
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def looks_like_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    cleaned = text.strip()
    if len(cleaned) < 4:
        return False
    # 简单检查是否包含中文或英文字符，若皆无则视为非文本
    return bool(re.search(r"[\u4e00-\u9fa5A-Za-z]", cleaned))


def summarize_batch(model, tokenizer, texts: List[str], max_source_length: int, max_target_length: int, device: str):
    tokenized = tokenizer(
        texts,
        max_length=max_source_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    with torch.no_grad():
        generated_ids = model.generate(
            **tokenized,
            max_length=max_target_length,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_single(text: str, model, tokenizer, device: str, max_source_length: int, max_target_length: int):
    if not looks_like_text(text):
        return "输入内容不是有效文本，请提供一段可阅读的中文或英文。"
    inputs = tokenizer([text], max_length=max_source_length, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=max_target_length,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def predict_dataset(
    dataset_file: str,
    model,
    tokenizer,
    device: str,
    max_source_length: int,
    max_target_length: int,
    batch_size: int,
) -> List[Tuple[str, str]]:
    ds = datasets.load_dataset("json", data_files={"predict": dataset_file})["predict"]
    contents = ds["content"]
    outputs: List[Tuple[str, str]] = []
    for idx in range(0, len(contents), batch_size):
        batch_texts = contents[idx : idx + batch_size]
        batch_summaries = summarize_batch(
            model=model,
            tokenizer=tokenizer,
            texts=batch_texts,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            device=device,
        )
        outputs.extend(zip(batch_texts, batch_summaries))
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize text or a dataset with a trained model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained checkpoint directory.")
    parser.add_argument("--text", type=str, help="Single text to summarize.")
    parser.add_argument("--dataset_file", type=str, help="Optional jsonl with 'content' field to summarize in batch.")
    parser.add_argument("--batch_mode", action="store_true", help="If set, write batch predictions to stdout in jsonl.")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for dataset prediction.")
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer, device = load_model(args.model_dir)

    if args.batch_mode and args.dataset_file:
        results = predict_dataset(
            dataset_file=args.dataset_file,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            batch_size=args.batch_size,
        )
        for content, summary in results:
            print(json.dumps({"content": content, "pred_summary": summary}, ensure_ascii=False))
        return

    if args.text:
        summary = predict_single(
            text=args.text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        )
        print(summary)
    else:
        print("请提供 --text 或 --dataset_file 进行摘要。")


if __name__ == "__main__":
    main()

