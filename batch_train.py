"""
批量训练脚本 - 依次训练所有三个模型
"""
import os
import sys
from config import Config
from train_with_hp_tuning import train_model, hyperparameter_search, set_seed
from data_processor import DataProcessor
import argparse


def batch_train_all_models(
    hp_search: bool = False,
    max_train_samples: int = None,
    max_valid_samples: int = None,
    custom_hyperparams: dict = None
):
    """
    批量训练所有模型
    
    Args:
        hp_search: 是否进行超参数搜索
        max_train_samples: 最大训练样本数
        max_valid_samples: 最大验证样本数
        custom_hyperparams: 自定义超参数
    """
    print("=" * 80)
    print("批量训练所有模型")
    print("=" * 80)
    
    # 设置随机种子
    set_seed(Config.SEED)
    
    results = []
    
    for model_key in Config.AVAILABLE_MODELS.keys():
        model_info = Config.AVAILABLE_MODELS[model_key]
        model_name = model_info["name"]
        
        print(f"\n{'=' * 80}")
        print(f"开始训练模型 [{model_key}]: {model_name}")
        print(f"{'=' * 80}")
        
        try:
            # 加载数据（每个模型都需要重新加载，因为tokenizer可能不同）
            print("\n加载数据...")
            processor = DataProcessor(model_name)
            train_data = processor.load_data(Config.TRAIN_FILE, max_samples=max_train_samples)
            valid_data = processor.load_data(Config.VALID_FILE, max_samples=max_valid_samples)
            
            print(f"训练集大小: {len(train_data)}")
            print(f"验证集大小: {len(valid_data)}")
            
            if hp_search:
                # 超参数搜索
                search_space = {
                    'batch_size': [4, 8],
                    'learning_rate': [1e-5, 3e-5, 5e-5],
                    'num_epochs': [2, 3],
                }
                if custom_hyperparams:
                    search_space.update(custom_hyperparams)
                
                search_result = hyperparameter_search(
                    model_name=model_name,
                    model_key=model_key,
                    train_data=train_data,
                    valid_data=valid_data,
                    search_space=search_space
                )
                results.append({
                    'model_key': model_key,
                    'model_name': model_name,
                    'type': 'hp_search',
                    'result': search_result
                })
            else:
                # 单次训练
                output_dir = os.path.join(Config.OUTPUT_DIR, f"{model_key}_finetuned")
                hyperparams = custom_hyperparams or {}
                
                training_info = train_model(
                    model_name=model_name,
                    model_key=model_key,
                    output_dir=output_dir,
                    train_data=train_data,
                    valid_data=valid_data,
                    hyperparams=hyperparams if hyperparams else None
                )
                results.append({
                    'model_key': model_key,
                    'model_name': model_name,
                    'type': 'single_train',
                    'result': training_info
                })
        
        except Exception as e:
            print(f"❌ 训练模型 {model_name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                'model_key': model_key,
                'model_name': model_name,
                'type': 'error',
                'error': str(e)
            })
            continue
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("批量训练完成")
    print("=" * 80)
    
    for result in results:
        print(f"\n模型 [{result['model_key']}]: {result['model_name']}")
        if result['type'] == 'hp_search':
            best_score = result['result'].get('best_score', 0)
            best_hp = result['result'].get('best_hyperparams', {})
            print(f"  类型: 超参数搜索")
            print(f"  最佳ROUGE-L: {best_score:.4f}")
            print(f"  最佳超参数: {best_hp}")
        elif result['type'] == 'single_train':
            best_rougeL = result['result'].get('best_rougeL', 0)
            print(f"  类型: 单次训练")
            print(f"  最佳ROUGE-L: {best_rougeL:.4f}")
        else:
            print(f"  类型: 错误")
            print(f"  错误信息: {result.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量训练所有模型')
    parser.add_argument('--hp_search', action='store_true', help='是否进行超参数搜索')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小（用于单次训练）')
    parser.add_argument('--learning_rate', type=float, default=None, help='学习率（用于单次训练）')
    parser.add_argument('--num_epochs', type=int, default=None, help='训练轮数（用于单次训练）')
    parser.add_argument('--max_train_samples', type=int, default=None, help='最大训练样本数')
    parser.add_argument('--max_valid_samples', type=int, default=None, help='最大验证样本数')
    
    args = parser.parse_args()
    
    custom_hyperparams = {}
    if args.batch_size:
        custom_hyperparams['batch_size'] = args.batch_size
    if args.learning_rate:
        custom_hyperparams['learning_rate'] = args.learning_rate
    if args.num_epochs:
        custom_hyperparams['num_epochs'] = args.num_epochs
    
    batch_train_all_models(
        hp_search=args.hp_search,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
        custom_hyperparams=custom_hyperparams if custom_hyperparams else None
    )

