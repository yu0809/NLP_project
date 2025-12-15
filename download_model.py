"""
模型下载脚本 - 下载所有三个预训练模型到本地缓存
模型会自动缓存到本地，后续使用时会优先从本地加载
支持使用国内镜像加速下载
"""
import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import Config

# 国内镜像配置
MIRROR_SITES = {
    "1": {
        "name": "hf-mirror.com (推荐)",
        "url": "https://hf-mirror.com",
        "env_var": "HF_ENDPOINT"
    },
    "2": {
        "name": "不使用镜像（直接连接Hugging Face）",
        "url": None,
        "env_var": None
    }
}


def setup_mirror(mirror_choice: str = None):
    """
    设置镜像源
    
    Args:
        mirror_choice: 镜像选择（1或2），如果为None则询问用户
    """
    if mirror_choice is None:
        print("\n选择下载镜像源:")
        for key, mirror in MIRROR_SITES.items():
            print(f"  [{key}] {mirror['name']}")
        
        choice = input("\n请选择镜像源 (1-2，默认1): ").strip() or "1"
    else:
        choice = mirror_choice
    
    if choice in MIRROR_SITES:
        mirror = MIRROR_SITES[choice]
        if mirror['url']:
            # 设置环境变量（多种方式确保生效）
            os.environ['HF_ENDPOINT'] = mirror['url']
            
            # 尝试使用 huggingface_hub 的配置
            try:
                import huggingface_hub
                # 设置端点
                if hasattr(huggingface_hub, 'constants'):
                    huggingface_hub.constants.ENDPOINT = mirror['url']
                # 或者使用 HfApi
                try:
                    from huggingface_hub import HfApi
                    # 这会使用环境变量中的 HF_ENDPOINT
                except:
                    pass
            except ImportError:
                print("   提示: 未安装 huggingface_hub，仅使用环境变量")
            except Exception as e:
                print(f"   警告: 设置 huggingface_hub 配置时出错: {str(e)}")
            
            print(f"\n✓ 已设置镜像源: {mirror['name']}")
            print(f"  镜像地址: {mirror['url']}")
            print(f"  环境变量 HF_ENDPOINT = {os.environ.get('HF_ENDPOINT')}")
            print(f"\n  注意: 如果仍然连接 huggingface.co，请尝试:")
            print(f"  1. 在终端中手动设置: export HF_ENDPOINT={mirror['url']}")
            print(f"  2. 然后重新运行此脚本")
            return mirror['url']
        else:
            print("\n✓ 使用直接连接（不使用镜像）")
            # 清除可能存在的镜像设置
            if 'HF_ENDPOINT' in os.environ:
                del os.environ['HF_ENDPOINT']
            return None
    else:
        print(f"⚠️  无效选择，使用默认镜像: {MIRROR_SITES['1']['name']}")
        os.environ['HF_ENDPOINT'] = MIRROR_SITES['1']['url']
        return MIRROR_SITES['1']['url']


def download_single_model(model_key: str, model_name: str, model_info: dict, mirror_url: str = None):
    """
    下载单个模型
    
    Args:
        model_key: 模型键
        model_name: 模型名称
        model_info: 模型信息
        mirror_url: 镜像URL，如果提供则使用镜像
    """
    print(f"\n{'=' * 80}")
    print(f"下载模型 [{model_key}]: {model_name}")
    print(f"描述: {model_info['description']}")
    print(f"大小: {model_info['size']}")
    if mirror_url:
        print(f"使用镜像: {mirror_url}")
    print("=" * 80)
    
    try:
        # 如果使用镜像，需要修改模型名称或使用镜像URL
        download_model_name = model_name
        
        # 对于 hf-mirror.com，需要将模型路径转换为镜像URL格式
        # 但 transformers 库可能不支持直接使用镜像URL
        # 所以我们需要确保环境变量已正确设置
        
        # 验证环境变量
        if mirror_url:
            current_endpoint = os.environ.get('HF_ENDPOINT', '')
            if current_endpoint != mirror_url:
                print(f"⚠️  警告: 环境变量 HF_ENDPOINT={current_endpoint}，期望 {mirror_url}")
                print(f"   重新设置环境变量...")
                os.environ['HF_ENDPOINT'] = mirror_url
        
        # 下载tokenizer
        print("\n1. 下载tokenizer...")
        print(f"   当前 HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置')}")
        
        # 使用镜像时，可能需要通过 huggingface_hub 设置
        if mirror_url:
            try:
                from huggingface_hub import snapshot_download
                # 先尝试使用镜像下载配置文件
                print(f"   尝试通过镜像下载...")
            except:
                pass
        
        tokenizer = AutoTokenizer.from_pretrained(download_model_name)
        print("   ✓ Tokenizer下载完成")
        
        # 下载模型
        print("\n2. 下载模型...")
        model = AutoModelForSeq2SeqLM.from_pretrained(download_model_name)
        print("   ✓ 模型下载完成")
        
        print(f"\n✓ 模型 [{model_key}] {model_name} 下载成功！")
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️  模型 [{model_key}] 下载已中断")
        return False
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ 模型 [{model_key}] 下载失败")
        print(f"   错误: {error_msg[:300]}...")  # 显示更多错误信息
        
        # 检查是否是连接问题
        if 'huggingface.co' in error_msg or 'timeout' in error_msg.lower():
            print("\n   检测到连接问题，可能的原因:")
            print("   1. 镜像配置未生效，尝试重新设置环境变量")
            print("   2. 网络连接问题")
            print("   3. 可以尝试不使用镜像: python download_model.py --no-mirror")
        
        # 检查是否是文件加载问题
        elif 'Unable to load weights' in error_msg or 'pytorch_model.bin' in error_msg:
            print("\n   检测到模型文件加载问题，可能的原因:")
            print("   1. 文件下载不完整或损坏")
            print("   2. 磁盘空间不足")
            print("   3. 文件权限问题")
            print("\n   解决方案:")
            print("   1. 删除损坏的缓存文件:")
            cache_path = f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}"
            print(f"      rm -rf {cache_path}")
            print("   2. 重新运行下载脚本")
            print("   3. 检查磁盘空间: df -h")
        
        # 检查是否是磁盘空间问题
        elif 'No space left' in error_msg or 'disk' in error_msg.lower():
            print("\n   检测到磁盘空间问题:")
            print("   1. 检查磁盘空间: df -h")
            print("   2. 清理不需要的文件")
            print("   3. 三个模型总共需要约7GB空间")
        
        return False


def download_all_models(use_mirror: bool = True, mirror_choice: str = None):
    """下载所有预训练模型"""
    print("=" * 80)
    print("模型下载工具 - 下载所有三个预训练模型")
    print("=" * 80)
    
    print("\n将下载以下模型:")
    total_size = 0
    for key, info in Config.AVAILABLE_MODELS.items():
        print(f"  [{key}] {info['name']} - {info['size']}")
        # 估算总大小（简单相加）
        size_str = info['size'].replace('~', '').replace('GB', '').replace('MB', '')
        try:
            if 'GB' in info['size']:
                total_size += float(size_str) * 1024  # 转换为MB
            else:
                total_size += float(size_str)
        except:
            pass
    
    print(f"\n预计总大小: ~{total_size/1024:.1f}GB")
    
    # 设置镜像
    if use_mirror:
        mirror_url = setup_mirror(mirror_choice)
    else:
        mirror_url = None
        print("\n使用直接连接（不使用镜像）")
    
    print("\n提示:")
    print("1. 模型会自动缓存到本地，后续使用时会优先从本地加载")
    print("2. 如果下载失败，可以尝试切换镜像源或使用VPN")
    print("-" * 80)
    
    confirm = input("\n是否开始下载所有模型? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消下载")
        return
    
    print("\n开始下载模型，请耐心等待...")
    print("这可能需要较长时间，取决于网络速度")
    print("模型会自动缓存到本地，后续使用时会优先从本地加载")
    print("-" * 80)
    
    results = {}
    success_count = 0
    fail_count = 0
    
    for model_key, model_info in Config.AVAILABLE_MODELS.items():
        model_name = model_info["name"]
        
        success = download_single_model(model_key, model_name, model_info, mirror_url)
        results[model_key] = {
            'model_name': model_name,
            'success': success
        }
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("下载完成汇总")
    print("=" * 80)
    
    for model_key, result in results.items():
        status = "✓ 成功" if result['success'] else "❌ 失败"
        print(f"[{model_key}] {result['model_name']}: {status}")
    
    print(f"\n成功: {success_count} 个模型")
    print(f"失败: {fail_count} 个模型")
    
    if success_count > 0:
        print("\n" + "=" * 80)
        print("✓ 模型已保存到本地缓存目录")
        print("后续使用时会自动从本地加载，无需再次下载")
        print("=" * 80)
        
        # 显示缓存路径
        try:
            from transformers import file_utils
            cache_dir = file_utils.default_cache_path
            print(f"\n模型缓存路径: {cache_dir}")
        except:
            pass
    
    if fail_count > 0:
        print("\n⚠️  部分模型下载失败，可以:")
        print("1. 检查网络连接")
        print("2. 尝试使用VPN")
        print("3. 重新运行此脚本继续下载")


def download_specific_model(model_key: str, use_mirror: bool = True, mirror_choice: str = None):
    """下载指定模型"""
    if model_key not in Config.AVAILABLE_MODELS:
        print(f"❌ 无效的模型键: {model_key}")
        print(f"可用模型键: {list(Config.AVAILABLE_MODELS.keys())}")
        return
    
    model_info = Config.AVAILABLE_MODELS[model_key]
    model_name = model_info["name"]
    
    print("=" * 80)
    print(f"下载模型 [{model_key}]: {model_name}")
    print("=" * 80)
    
    # 设置镜像
    if use_mirror:
        setup_mirror(mirror_choice)
    
    confirm = input(f"\n是否下载模型 [{model_key}]? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消下载")
        return
    
    mirror_url = setup_mirror(mirror_choice) if use_mirror else None
    download_single_model(model_key, model_name, model_info, mirror_url)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='下载预训练模型（支持国内镜像加速）')
    parser.add_argument('--model_key', type=str, default=None, 
                       help='模型键（1-3），如果为None则下载所有模型')
    parser.add_argument('--mirror', type=str, default=None,
                       help='镜像选择（1=使用hf-mirror.com，2=不使用镜像），默认1')
    parser.add_argument('--no-mirror', action='store_true',
                       help='不使用镜像（直接连接）')
    
    args = parser.parse_args()
    
    # 确定是否使用镜像
    use_mirror = not args.no_mirror
    mirror_choice = args.mirror or ("1" if use_mirror else "2")
    
    if args.model_key:
        download_specific_model(args.model_key, use_mirror, mirror_choice)
    else:
        download_all_models(use_mirror, mirror_choice)

