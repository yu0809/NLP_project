"""
模型比较脚本 - 使用多个模型对同一文本生成摘要并比较
"""
import os
import time
from inference import Summarizer
from config import Config


def compare_models(text: str, model_keys: list = None):
    """
    比较多个模型的摘要生成效果
    
    Args:
        text: 要摘要的文本
        model_keys: 要比较的模型键列表，如果为None则比较所有模型
    """
    print("=" * 80)
    print("模型比较测试")
    print("=" * 80)
    
    if model_keys is None:
        model_keys = list(Config.AVAILABLE_MODELS.keys())
    
    print(f"\n测试文本（前100字符）: {text[:100]}...")
    print(f"\n将比较 {len(model_keys)} 个模型\n")
    print("-" * 80)
    
    results = []
    
    for key in model_keys:
        if key not in Config.AVAILABLE_MODELS:
            print(f"⚠️  跳过无效的模型键: {key}")
            continue
        
        model_info = Config.AVAILABLE_MODELS[key]
        model_name = model_info["name"]
        description = model_info["description"]
        
        print(f"\n[{key}] {model_name}")
        print(f"    描述: {description}")
        print("    正在加载模型...")
        
        try:
            start_time = time.time()
            summarizer = Summarizer(model_name)
            load_time = time.time() - start_time
            
            print(f"    模型加载时间: {load_time:.2f}秒")
            print("    正在生成摘要...")
            
            gen_start = time.time()
            summary = summarizer.generate_summary(text)
            gen_time = time.time() - gen_start
            
            results.append({
                "key": key,
                "model_name": model_name,
                "description": description,
                "summary": summary,
                "load_time": load_time,
                "generation_time": gen_time,
                "summary_length": len(summary)
            })
            
            print(f"    生成时间: {gen_time:.2f}秒")
            print(f"    摘要长度: {len(summary)} 字符")
            print(f"    摘要: {summary[:80]}..." if len(summary) > 80 else f"    摘要: {summary}")
            
        except Exception as e:
            print(f"    ❌ 错误: {str(e)}")
            results.append({
                "key": key,
                "model_name": model_name,
                "description": description,
                "summary": f"错误: {str(e)}",
                "load_time": 0,
                "generation_time": 0,
                "summary_length": 0
            })
        
        print("-" * 80)
    
    # 打印比较结果
    print("\n" + "=" * 80)
    print("比较结果汇总")
    print("=" * 80)
    
    print(f"\n{'模型':<30} {'加载时间':<12} {'生成时间':<12} {'摘要长度':<10} {'摘要预览'}")
    print("-" * 80)
    
    for result in results:
        model_display = result["model_name"].split("/")[-1][:28]
        load_t = f"{result['load_time']:.2f}s"
        gen_t = f"{result['generation_time']:.2f}s"
        length = str(result['summary_length'])
        preview = result['summary'][:30] + "..." if len(result['summary']) > 30 else result['summary']
        
        print(f"{model_display:<30} {load_t:<12} {gen_t:<12} {length:<10} {preview}")
    
    print("\n" + "=" * 80)
    print("详细摘要对比")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['model_name']}")
        print(f"描述: {result['description']}")
        print(f"摘要: {result['summary']}")
        print("-" * 80)
    
    return results


def interactive_compare():
    """交互式模型比较"""
    print("=" * 80)
    print("模型比较工具")
    print("=" * 80)
    
    # 显示可用模型
    print("\n可用模型列表:")
    print("-" * 80)
    for key, info in Config.AVAILABLE_MODELS.items():
        print(f"[{key}] {info['name']}")
        print(f"     {info['description']} ({info['size']})")
    
    # 选择模型
    print("\n" + "-" * 80)
    print("请选择要比较的模型（输入模型编号，用逗号分隔，如: 1,2,3）")
    print("输入 'all' 比较所有模型")
    print("输入 'quit' 退出")
    
    while True:
        choice = input("\n请选择: ").strip().lower()
        
        if choice == 'quit':
            print("退出")
            return
        
        if choice == 'all':
            model_keys = list(Config.AVAILABLE_MODELS.keys())
        else:
            model_keys = [k.strip() for k in choice.split(',')]
            # 验证键是否有效
            invalid_keys = [k for k in model_keys if k not in Config.AVAILABLE_MODELS]
            if invalid_keys:
                print(f"❌ 无效的模型键: {', '.join(invalid_keys)}")
                continue
        
        # 输入文本
        print("\n" + "-" * 80)
        print("请输入要摘要的文本（输入完成后按回车，输入多行文本时输入 'END' 结束）:")
        print("-" * 80)
        
        lines = []
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
        
        text = '\n'.join(lines).strip()
        
        if not text:
            print("❌ 输入文本为空")
            continue
        
        if len(text) < 10:
            print("❌ 文本过短，请输入至少10个字符")
            continue
        
        # 执行比较
        compare_models(text, model_keys)
        
        # 询问是否继续
        print("\n是否继续比较其他模型？(y/n): ", end='')
        if input().strip().lower() != 'y':
            break


if __name__ == "__main__":
    # 示例文本
    example_text = """人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。"""
    
    import sys
    if len(sys.argv) > 1:
        # 命令行模式
        if sys.argv[1] == '--interactive' or sys.argv[1] == '-i':
            interactive_compare()
        elif sys.argv[1] == '--models' or sys.argv[1] == '-m':
            # 指定模型列表
            if len(sys.argv) < 3:
                print("错误: 请指定模型键，如: --models 1,2,3")
                sys.exit(1)
            model_keys = [k.strip() for k in sys.argv[2].split(',')]
            if len(sys.argv) > 3:
                text = ' '.join(sys.argv[3:])
            else:
                text = example_text
            compare_models(text, model_keys)
        else:
            # 使用命令行参数作为文本
            text = ' '.join(sys.argv[1:])
            compare_models(text)
    else:
        # 使用示例文本
        print("使用示例文本进行模型比较...")
        print("提示:")
        print("  - 使用 --interactive 或 -i 参数进入交互模式")
        print("  - 直接提供文本作为参数: python compare_models.py '你的文本内容'")
        print("  - 指定模型: python compare_models.py --models 1,2,3 '你的文本内容'")
        print("\n")
        compare_models(example_text, ["1", "2", "3"])  # 默认比较所有三个模型

