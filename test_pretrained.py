"""
测试预训练模型脚本 - 直接使用预训练模型在测试集上评估（不微调）
"""
import os
import json
from datetime import datetime
from inference import Summarizer
from config import Config
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from data_processor import DataProcessor


def test_pretrained_model(model_key: str, test_file: str = None, max_samples: int = None):
    """
    测试单个预训练模型（不微调）
    
    Args:
        model_key: 模型键（1-3）
        test_file: 测试文件路径
        max_samples: 最大测试样本数
    """
    if model_key not in Config.AVAILABLE_MODELS:
        print(f"❌ 无效的模型键: {model_key}")
        return None
    
    model_info = Config.AVAILABLE_MODELS[model_key]
    model_name = model_info["name"]
    
    print("=" * 80)
    print(f"测试预训练模型: {model_name}")
    print(f"描述: {model_info['description']}")
    print("=" * 80)
    
    # 初始化模型
    print("\n正在加载预训练模型...")
    try:
        summarizer = Summarizer(model_name)
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return None
    
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
    
    # 用于BERTScore的列表
    predictions_list = []
    references_list = []
    
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
            
            # 保存用于BERTScore
            predictions_list.append(prediction)
            references_list.append(reference)
        
        except Exception as e:
            print(f"处理第 {i} 个样本时出错: {str(e)}")
            continue
    
    # 计算平均ROUGE分数
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    
    # 计算BERTScore
    print("\n正在计算BERTScore（这可能需要一些时间）...")
    try:
        # BERTScore返回P, R, F1三个分数
        P, R, F1 = bert_score(
            predictions_list,
            references_list,
            lang='zh',  # 中文
            verbose=False
        )
        avg_bertscore_precision = P.mean().item()
        avg_bertscore_recall = R.mean().item()
        avg_bertscore_f1 = F1.mean().item()
    except Exception as e:
        print(f"⚠️  BERTScore计算失败: {str(e)}")
        print("   继续使用ROUGE分数...")
        avg_bertscore_precision = 0.0
        avg_bertscore_recall = 0.0
        avg_bertscore_f1 = 0.0
    
    result = {
        'model_key': model_key,
        'model_name': model_name,
        'description': model_info['description'],
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'bertscore_precision': avg_bertscore_precision,
        'bertscore_recall': avg_bertscore_recall,
        'bertscore_f1': avg_bertscore_f1,
        'num_samples': len(rouge1_scores),
        'test_file': test_file,
        'timestamp': datetime.now().isoformat()
    }
    
    # 打印结果
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)
    print(f"模型: {model_name}")
    print(f"\nROUGE指标:")
    print(f"  ROUGE-1: {avg_rouge1:.4f}")
    print(f"  ROUGE-2: {avg_rouge2:.4f}")
    print(f"  ROUGE-L: {avg_rougeL:.4f}")
    print(f"\nBERTScore指标:")
    print(f"  Precision: {avg_bertscore_precision:.4f}")
    print(f"  Recall: {avg_bertscore_recall:.4f}")
    print(f"  F1: {avg_bertscore_f1:.4f}")
    print(f"\n测试样本数: {len(rouge1_scores)}")
    print("=" * 80)
    
    return result


def test_all_pretrained_models(test_file: str = None, max_samples: int = None):
    """
    测试所有预训练模型
    
    Args:
        test_file: 测试文件路径
        max_samples: 最大测试样本数
    """
    print("=" * 80)
    print("测试所有预训练模型（不微调）")
    print("=" * 80)
    
    results = []
    
    for model_key in Config.AVAILABLE_MODELS.keys():
        result = test_pretrained_model(model_key, test_file, max_samples)
        if result:
            results.append(result)
        print("\n")
    
    # 保存结果
    output_file = "pretrained_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印汇总
    print("=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    # ROUGE指标表格
    print(f"\n{'模型':<35} {'ROUGE-1':<12} {'ROUGE-2':<12} {'ROUGE-L':<12}")
    print("-" * 80)
    
    # 按ROUGE-L排序
    sorted_results = sorted(results, key=lambda x: x['rougeL'], reverse=True)
    
    for result in sorted_results:
        model_display = result['model_name'].split("/")[-1][:33]
        r1 = f"{result['rouge1']:.4f}"
        r2 = f"{result['rouge2']:.4f}"
        rL = f"{result['rougeL']:.4f}"
        print(f"{model_display:<35} {r1:<12} {r2:<12} {rL:<12}")
    
    # BERTScore指标表格
    print(f"\n{'模型':<35} {'BERTScore-P':<15} {'BERTScore-R':<15} {'BERTScore-F1':<15}")
    print("-" * 80)
    
    for result in sorted_results:
        model_display = result['model_name'].split("/")[-1][:33]
        bs_p = f"{result.get('bertscore_precision', 0):.4f}"
        bs_r = f"{result.get('bertscore_recall', 0):.4f}"
        bs_f1 = f"{result.get('bertscore_f1', 0):.4f}"
        print(f"{model_display:<35} {bs_p:<15} {bs_r:<15} {bs_f1:<15}")
    
    print(f"\n详细结果已保存到: {output_file}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试预训练模型（不微调）')
    parser.add_argument('--model_key', type=str, default=None, help='模型键（1-3），如果为None则测试所有模型')
    parser.add_argument('--test_file', type=str, default=None, help='测试文件路径')
    parser.add_argument('--max_samples', type=int, default=None, help='最大测试样本数')
    
    args = parser.parse_args()
    
    if args.model_key:
        test_pretrained_model(args.model_key, args.test_file, args.max_samples)
    else:
        test_all_pretrained_models(args.test_file, args.max_samples)

