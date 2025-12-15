"""
使用示例脚本
"""
from inference import Summarizer
from config import Config
import os

def example_usage():
    """使用示例"""
    print("=" * 60)
    print("自动摘要系统使用示例")
    print("=" * 60)
    
    # 初始化摘要器
    print("\n1. 初始化摘要器...")
    model_path = None
    if os.path.exists(os.path.join(Config.OUTPUT_DIR, "final_model")):
        model_path = os.path.join(Config.OUTPUT_DIR, "final_model")
        print(f"   使用已训练的模型: {model_path}")
    else:
        print(f"   使用预训练模型: {Config.MODEL_NAME}")
    
    summarizer = Summarizer(model_path)
    
    # 示例文本
    example_texts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。",
        
        "深度学习是机器学习的一个子领域，它基于人工神经网络进行表征学习。深度学习模型可以自动从数据中学习特征表示，而不需要人工设计特征。深度学习的成功应用包括图像识别、语音识别、自然语言处理等领域。",
        
        "自然语言处理是计算机科学和人工智能的一个分支，它研究如何让计算机理解、处理和生成人类语言。自然语言处理的应用包括机器翻译、情感分析、文本摘要、问答系统等。"
    ]
    
    print("\n2. 生成摘要示例...")
    print("-" * 60)
    
    for i, text in enumerate(example_texts, 1):
        print(f"\n示例 {i}:")
        print(f"原文: {text[:50]}...")
        
        try:
            summary = summarizer.generate_summary(text)
            print(f"摘要: {summary}")
        except Exception as e:
            print(f"错误: {str(e)}")
        
        print("-" * 60)
    
    # 测试输入验证
    print("\n3. 测试输入验证...")
    print("-" * 60)
    
    invalid_inputs = [
        "",  # 空输入
        "123456",  # 纯数字
        "   ",  # 空白字符
        "abc",  # 过短文本
    ]
    
    for invalid_input in invalid_inputs:
        print(f"\n测试输入: '{invalid_input}'")
        try:
            summary = summarizer.generate_summary(invalid_input)
            print(f"摘要: {summary}")
        except ValueError as e:
            print(f"✓ 正确检测到无效输入: {str(e)}")
        except Exception as e:
            print(f"错误: {str(e)}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()

