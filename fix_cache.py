"""
修复模型缓存脚本 - 清理损坏的模型文件并重新下载
"""
import os
import shutil
from config import Config


def clear_model_cache(model_key: str = None):
    """
    清理指定模型或所有模型的缓存
    
    Args:
        model_key: 模型键（1-3），如果为None则清理所有模型
    """
    import os
    from pathlib import Path
    
    # 获取缓存目录
    cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    
    if model_key:
        if model_key not in Config.AVAILABLE_MODELS:
            print(f"❌ 无效的模型键: {model_key}")
            return
        
        model_name = Config.AVAILABLE_MODELS[model_key]["name"]
        model_cache_dir = cache_base / f"models--{model_name.replace('/', '--')}"
        
        if model_cache_dir.exists():
            print(f"正在清理模型 [{model_key}] {model_name} 的缓存...")
            print(f"缓存路径: {model_cache_dir}")
            try:
                shutil.rmtree(model_cache_dir)
                print(f"✓ 已清理模型 [{model_key}] 的缓存")
            except Exception as e:
                print(f"❌ 清理失败: {str(e)}")
        else:
            print(f"模型 [{model_key}] 的缓存不存在")
    else:
        # 清理所有模型
        print("正在清理所有模型的缓存...")
        print(f"缓存基础目录: {cache_base}")
        
        for model_key, model_info in Config.AVAILABLE_MODELS.items():
            model_name = model_info["name"]
            model_cache_dir = cache_base / f"models--{model_name.replace('/', '--')}"
            
            if model_cache_dir.exists():
                print(f"\n清理模型 [{model_key}] {model_name}...")
                try:
                    shutil.rmtree(model_cache_dir)
                    print(f"  ✓ 已清理")
                except Exception as e:
                    print(f"  ❌ 清理失败: {str(e)}")
            else:
                print(f"\n模型 [{model_key}] 的缓存不存在，跳过")
        
        print("\n✓ 所有模型缓存清理完成")


def check_disk_space():
    """检查磁盘空间"""
    import shutil
    
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    # 检查缓存目录所在磁盘的空间
    stat = shutil.disk_usage(cache_dir if os.path.exists(cache_dir) else os.path.expanduser("~"))
    
    total_gb = stat.total / (1024**3)
    used_gb = stat.used / (1024**3)
    free_gb = stat.free / (1024**3)
    
    print("=" * 60)
    print("磁盘空间检查")
    print("=" * 60)
    print(f"总空间: {total_gb:.2f} GB")
    print(f"已使用: {used_gb:.2f} GB")
    print(f"可用空间: {free_gb:.2f} GB")
    print(f"缓存目录: {cache_dir}")
    
    if free_gb < 10:
        print("\n⚠️  警告: 可用空间不足10GB，可能影响模型下载")
        print("   建议清理一些文件后再下载")
    else:
        print("\n✓ 磁盘空间充足")
    
    return free_gb


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='修复模型缓存')
    parser.add_argument('--model_key', type=str, default=None,
                       help='模型键（1-3），如果为None则清理所有模型')
    parser.add_argument('--check-space', action='store_true',
                       help='检查磁盘空间')
    
    args = parser.parse_args()
    
    if args.check_space:
        check_disk_space()
    else:
        print("=" * 60)
        print("模型缓存清理工具")
        print("=" * 60)
        print("\n此工具将清理损坏或不完整的模型缓存文件")
        print("清理后需要重新下载模型")
        print("-" * 60)
        
        if args.model_key:
            confirm = input(f"\n是否清理模型 [{args.model_key}] 的缓存? (y/n): ").strip().lower()
        else:
            confirm = input("\n是否清理所有模型的缓存? (y/n): ").strip().lower()
        
        if confirm == 'y':
            clear_model_cache(args.model_key)
            print("\n" + "=" * 60)
            print("清理完成！")
            print("现在可以重新运行下载脚本: python download_model.py")
            print("=" * 60)
        else:
            print("已取消")

