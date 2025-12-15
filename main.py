"""
主程序 - 自动摘要系统
"""
import os
import sys
from inference import Summarizer
from config import Config


def print_banner():
    """打印欢迎信息"""
    print("=" * 60)
    print(" " * 15 + "中文自动摘要系统")
    print("=" * 60)
    print()


def print_menu():
    """打印菜单"""
    print("\n请选择操作：")
    print("1. 输入文本生成摘要")
    print("2. 从文件读取文本生成摘要")
    print("3. 批量处理文件")
    print("4. 切换模型")
    print("5. 模型比较")
    print("6. 退出")
    print("-" * 60)


def validate_input(text: str):
    """
    验证输入是否为有效文本
    
    Args:
        text: 输入文本
        
    Returns:
        (是否有效, 错误信息)
    """
    if not text:
        return False, "输入为空，请输入文本内容"
    
    if not isinstance(text, str):
        return False, "输入不是文本类型，请输入字符串"
    
    # 检查是否包含至少一个中文字符或英文字母
    import re
    has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', text))
    has_english = bool(re.search(r'[a-zA-Z]', text))
    
    if not (has_chinese or has_english):
        return False, "输入不包含有效的中文或英文文本，请提供包含文字的文本内容"
    
    # 文本长度检查
    if len(text.strip()) < 10:
        return False, "输入文本过短（少于10个字符），请输入更长的文本"
    
    return True, ""


def single_text_mode(summarizer: Summarizer):
    """单文本模式"""
    print("\n" + "=" * 60)
    print("单文本摘要模式")
    print("=" * 60)
    print("\n请输入要摘要的文本（输入完成后按回车，输入'quit'退出）：")
    print("-" * 60)
    
    while True:
        try:
            text = input("\n输入文本: ").strip()
            
            if text.lower() == 'quit':
                break
            
            if not text:
                print("输入为空，请重新输入")
                continue
            
            # 验证输入
            is_valid, error_msg = validate_input(text)
            if not is_valid:
                print(f"❌ 输入验证失败: {error_msg}")
                continue
            
            # 生成摘要
            print("\n正在生成摘要...")
            try:
                summary = summarizer.generate_summary(text)
                print("\n" + "=" * 60)
                print("生成的摘要：")
                print("=" * 60)
                print(summary)
                print("=" * 60)
            except Exception as e:
                print(f"❌ 生成摘要时出错: {str(e)}")
        
        except KeyboardInterrupt:
            print("\n\n操作已取消")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")


def file_mode(summarizer: Summarizer):
    """文件模式"""
    print("\n" + "=" * 60)
    print("文件摘要模式")
    print("=" * 60)
    
    file_path = input("\n请输入文件路径: ").strip()
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            print("❌ 文件为空")
            return
        
        # 验证输入
        is_valid, error_msg = validate_input(text)
        if not is_valid:
            print(f"❌ 输入验证失败: {error_msg}")
            return
        
        # 生成摘要
        print("\n正在生成摘要...")
        try:
            summary = summarizer.generate_summary(text)
            print("\n" + "=" * 60)
            print("生成的摘要：")
            print("=" * 60)
            print(summary)
            print("=" * 60)
            
            # 询问是否保存
            save = input("\n是否保存摘要到文件？(y/n): ").strip().lower()
            if save == 'y':
                output_path = file_path + ".summary.txt"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("原文:\n")
                    f.write(text)
                    f.write("\n\n" + "=" * 60 + "\n\n")
                    f.write("摘要:\n")
                    f.write(summary)
                print(f"摘要已保存到: {output_path}")
        
        except Exception as e:
            print(f"❌ 生成摘要时出错: {str(e)}")
    
    except Exception as e:
        print(f"❌ 读取文件时出错: {str(e)}")


def batch_mode(summarizer: Summarizer):
    """批量处理模式"""
    print("\n" + "=" * 60)
    print("批量处理模式")
    print("=" * 60)
    
    input_dir = input("\n请输入包含文本文件的目录路径: ").strip()
    
    if not os.path.isdir(input_dir):
        print(f"❌ 目录不存在: {input_dir}")
        return
    
    # 查找所有文本文件
    import glob
    text_files = []
    for ext in ['*.txt', '*.md', '*.csv']:
        text_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not text_files:
        print(f"❌ 目录中没有找到文本文件")
        return
    
    print(f"\n找到 {len(text_files)} 个文件")
    output_dir = os.path.join(input_dir, "summaries")
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for i, file_path in enumerate(text_files, 1):
        print(f"\n处理文件 {i}/{len(text_files)}: {os.path.basename(file_path)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # 验证输入
            is_valid, error_msg = validate_input(text)
            if not is_valid:
                print(f"  ⚠️  跳过: {error_msg}")
                fail_count += 1
                continue
            
            # 生成摘要
            summary = summarizer.generate_summary(text)
            
            # 保存摘要
            output_path = os.path.join(
                output_dir,
                os.path.basename(file_path) + ".summary.txt"
            )
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("原文:\n")
                f.write(text)
                f.write("\n\n" + "=" * 60 + "\n\n")
                f.write("摘要:\n")
                f.write(summary)
            
            print(f"  ✅ 完成，摘要已保存")
            success_count += 1
        
        except Exception as e:
            print(f"  ❌ 处理失败: {str(e)}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"批量处理完成！")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"摘要保存在: {output_dir}")
    print("=" * 60)


def select_model():
    """选择模型"""
    print("\n" + "=" * 60)
    print("模型选择")
    print("=" * 60)
    print("\n可用模型列表:")
    print("-" * 60)
    for key, info in Config.AVAILABLE_MODELS.items():
        print(f"[{key}] {info['name']}")
        print(f"     {info['description']} ({info['size']})")
    print("[0] 使用已训练的模型（如果存在）")
    print("-" * 60)
    
    while True:
        choice = input("\n请选择模型 (0-3): ").strip()
        
        if choice == '0':
            model_path = os.path.join(Config.OUTPUT_DIR, "final_model")
            if os.path.exists(model_path):
                return model_path
            else:
                print("❌ 未找到已训练的模型")
                continue
        
        if choice in Config.AVAILABLE_MODELS:
            return Config.AVAILABLE_MODELS[choice]["name"]
        else:
            print("❌ 无效的选择，请输入 0-3")


def model_comparison_mode():
    """模型比较模式"""
    print("\n" + "=" * 60)
    print("模型比较模式")
    print("=" * 60)
    print("\n提示: 此功能将使用多个模型对同一文本生成摘要并比较")
    print("建议使用专门的比较脚本: python compare_models.py")
    print("-" * 60)
    
    use_script = input("\n是否使用专门的比较脚本？(y/n): ").strip().lower()
    if use_script == 'y':
        print("\n正在启动模型比较脚本...")
        import subprocess
        subprocess.run([sys.executable, "compare_models.py", "--interactive"])
    else:
        print("\n请手动运行: python compare_models.py --interactive")


def main():
    """主函数"""
    print_banner()
    
    # 初始化模型
    current_model = None
    summarizer = None
    
    def init_model(model_path_or_name):
        """初始化模型"""
        nonlocal summarizer, current_model
        print(f"\n正在初始化模型，请稍候...")
        try:
            summarizer = Summarizer(model_path_or_name)
            current_model = model_path_or_name
            print("✅ 模型初始化成功！\n")
            return True
        except Exception as e:
            print(f"❌ 模型初始化失败: {str(e)}")
            print("请检查模型路径或网络连接")
            return False
    
    # 检查是否有已训练的模型
    model_path = None
    if os.path.exists(os.path.join(Config.OUTPUT_DIR, "final_model")):
        model_path = os.path.join(Config.OUTPUT_DIR, "final_model")
        print(f"检测到已训练的模型: {model_path}")
        print("提示: 可以在菜单中切换其他预训练模型")
    else:
        print(f"未找到已训练的模型")
        print("提示: 运行 train.py 可以训练自己的模型")
    
    # 选择初始模型
    print("\n请选择初始模型:")
    print("[1] 使用默认模型 (google/mt5-base)")
    print("[2] 手动选择模型")
    if model_path:
        print("[3] 使用已训练的模型")
    
    init_choice = input("请选择 (1-3): ").strip()
    
    if init_choice == '2':
        selected_model = select_model()
        if not init_model(selected_model):
            sys.exit(1)
    elif init_choice == '3' and model_path:
        if not init_model(model_path):
            sys.exit(1)
    else:
        # 使用默认模型
        if not init_model(Config.MODEL_NAME):
            sys.exit(1)
    
    # 主循环
    while True:
        print_menu()
        try:
            choice = input("请选择 (1-6): ").strip()
            
            if choice == '1':
                if summarizer:
                    single_text_mode(summarizer)
                else:
                    print("❌ 模型未初始化")
            elif choice == '2':
                if summarizer:
                    file_mode(summarizer)
                else:
                    print("❌ 模型未初始化")
            elif choice == '3':
                if summarizer:
                    batch_mode(summarizer)
                else:
                    print("❌ 模型未初始化")
            elif choice == '4':
                # 切换模型
                selected_model = select_model()
                if init_model(selected_model):
                    print(f"✅ 已切换到模型: {current_model}")
            elif choice == '5':
                # 模型比较
                model_comparison_mode()
            elif choice == '6':
                print("\n感谢使用！再见！")
                break
            else:
                print("❌ 无效的选择，请输入 1-6")
        
        except KeyboardInterrupt:
            print("\n\n程序已退出")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")


if __name__ == "__main__":
    main()

