"""
模型评估脚本 - 支持单模型和多模型比较评估
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data_processor import DataProcessor
from inference import Summarizer
from config import Config
from rouge_score import rouge_scorer
import pandas as pd
import json


def evaluate_model(model_path: str = None, test_file: str = None, max_samples: int = 100):
    """
    评估模型性能
    
    Args:
        model_path: 模型路径
        test_file: 测试文件路径
        max_samples: 最大评估样本数
    """
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    
    # 初始化摘要器
    model_path = model_path or os.path.join(Config.OUTPUT_DIR, "final_model")
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        print("使用预训练模型进行评估...")
        model_path = None
    
    summarizer = Summarizer(model_path)
    
    # 加载测试数据
    test_file = test_file or Config.TEST_FILE
    print(f"\n加载测试数据: {test_file}")
    processor = DataProcessor()
    test_data = processor.load_data(test_file, max_samples=max_samples)
    
    print(f"测试样本数: {len(test_data)}")
    
    # 初始化ROUGE评估器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    print("\n开始评估...")
    print("-" * 60)
    
    for i, (reference, source) in enumerate(test_data, 1):
        if i % 10 == 0:
            print(f"处理进度: {i}/{len(test_data)}")
        
        try:
            # 生成摘要
            prediction = summarizer.generate_summary(source)
            
            # 计算ROUGE分数
            scores = scorer.score(reference, prediction)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        except Exception as e:
            print(f"处理第 {i} 个样本时出错: {str(e)}")
            continue
    
    # 计算平均分数
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougeL:.4f}")
    print("=" * 60)
    
    return {
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL
    }


def compare_models_evaluation(model_keys: list = None, test_file: str = None, max_samples: int = 100):
    """
    比较多个模型的评估结果
    
    Args:
        model_keys: 要评估的模型键列表，如果为None则评估所有模型
        test_file: 测试文件路径
        max_samples: 最大评估样本数
    """
    print("=" * 80)
    print("多模型评估比较")
    print("=" * 80)
    
    if model_keys is None:
        model_keys = list(Config.AVAILABLE_MODELS.keys())
    
    test_file = test_file or Config.TEST_FILE
    
    # 加载测试数据
    print(f"\n加载测试数据: {test_file}")
    processor = DataProcessor()
    test_data = processor.load_data(test_file, max_samples=max_samples)
    print(f"测试样本数: {len(test_data)}")
    
    # 初始化ROUGE评估器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    results = []
    
    for key in model_keys:
        if key not in Config.AVAILABLE_MODELS:
            print(f"⚠️  跳过无效的模型键: {key}")
            continue
        
        model_info = Config.AVAILABLE_MODELS[key]
        model_name = model_info["name"]
        description = model_info["description"]
        
        print(f"\n{'=' * 80}")
        print(f"评估模型 [{key}]: {model_name}")
        print(f"描述: {description}")
        print(f"{'=' * 80}")
        
        try:
            # 初始化模型
            print("正在加载模型...")
            summarizer = Summarizer(model_name)
            
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            print("\n开始评估...")
            print("-" * 80)
            
            for i, (reference, source) in enumerate(test_data, 1):
                if i % 10 == 0:
                    print(f"处理进度: {i}/{len(test_data)}")
                
                try:
                    # 生成摘要
                    prediction = summarizer.generate_summary(source)
                    
                    # 计算ROUGE分数
                    scores = scorer.score(reference, prediction)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
                
                except Exception as e:
                    print(f"处理第 {i} 个样本时出错: {str(e)}")
                    continue
            
            # 计算平均分数
            avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
            avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
            avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
            
            result = {
                'key': key,
                'model_name': model_name,
                'description': description,
                'rouge1': avg_rouge1,
                'rouge2': avg_rouge2,
                'rougeL': avg_rougeL,
                'num_samples': len(rouge1_scores)
            }
            
            results.append(result)
            
            print(f"\n评估完成:")
            print(f"  ROUGE-1: {avg_rouge1:.4f}")
            print(f"  ROUGE-2: {avg_rouge2:.4f}")
            print(f"  ROUGE-L: {avg_rougeL:.4f}")
        
        except Exception as e:
            print(f"❌ 评估模型 {model_name} 时出错: {str(e)}")
            results.append({
                'key': key,
                'model_name': model_name,
                'description': description,
                'rouge1': 0,
                'rouge2': 0,
                'rougeL': 0,
                'error': str(e)
            })
    
    # 打印比较结果
    print("\n" + "=" * 80)
    print("评估结果汇总")
    print("=" * 80)
    
    print(f"\n{'模型':<35} {'ROUGE-1':<12} {'ROUGE-2':<12} {'ROUGE-L':<12} {'样本数'}")
    print("-" * 80)
    
    # 按ROUGE-L排序
    sorted_results = sorted(results, key=lambda x: x['rougeL'], reverse=True)
    
    for result in sorted_results:
        model_display = result['model_name'].split("/")[-1][:33]
        r1 = f"{result['rouge1']:.4f}"
        r2 = f"{result['rouge2']:.4f}"
        rL = f"{result['rougeL']:.4f}"
        num = str(result.get('num_samples', 0))
        
        print(f"{model_display:<35} {r1:<12} {r2:<12} {rL:<12} {num}")
    
    # 保存结果到文件
    output_file = "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='评估模型性能')
    parser.add_argument('--model_path', type=str, default=None, help='模型路径或模型名称')
    parser.add_argument('--model_key', type=str, default=None, help='模型键（1-3）')
    parser.add_argument('--compare', action='store_true', help='比较多个模型')
    parser.add_argument('--models', type=str, default=None, help='要比较的模型键列表，用逗号分隔（如: 1,2,3）')
    parser.add_argument('--test_file', type=str, default=None, help='测试文件路径')
    parser.add_argument('--max_samples', type=int, default=100, help='最大评估样本数')
    
    args = parser.parse_args()
    
    if args.compare:
        # 多模型比较模式
        model_keys = None
        if args.models:
            model_keys = [k.strip() for k in args.models.split(',')]
        compare_models_evaluation(
            model_keys=model_keys,
            test_file=args.test_file,
            max_samples=args.max_samples
        )
    else:
        # 单模型评估模式
        model_path = args.model_path
        if args.model_key:
            if args.model_key in Config.AVAILABLE_MODELS:
                model_path = Config.AVAILABLE_MODELS[args.model_key]["name"]
            else:
                print(f"❌ 无效的模型键: {args.model_key}")
                print("可用模型键:", list(Config.AVAILABLE_MODELS.keys()))
                exit(1)
        
        evaluate_model(
            model_path=model_path,
            test_file=args.test_file,
            max_samples=args.max_samples
        )

