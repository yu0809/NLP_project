"""
推理模块
"""
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import Config
from data_processor import DataProcessor


class Summarizer:
    """摘要生成器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化摘要生成器
        
        Args:
            model_path: 模型路径，如果为None则使用预训练模型
        """
        # 自动选择最佳设备（优先：CUDA > MPS > CPU）
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Apple Silicon GPU
        else:
            self.device = torch.device("cpu")
        
        model_path = model_path or Config.MODEL_NAME
        
        print(f"正在加载模型: {model_path}")
        print(f"使用设备: {self.device}")
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        except:
            # 如果指定路径不存在，使用预训练模型
            print(f"警告: 无法加载 {model_path}，使用预训练模型 {Config.MODEL_NAME}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
        
        self.model.to(self.device)
        self.model.eval()
        
        print("模型加载完成！")
    
    def is_valid_text(self, text: str) -> bool:
        """
        检查输入是否为有效文本
        
        Args:
            text: 输入文本
            
        Returns:
            是否为有效文本
        """
        if not text or not isinstance(text, str):
            return False
        
        # 检查是否包含至少一个中文字符或英文字母
        has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', text))
        has_english = bool(re.search(r'[a-zA-Z]', text))
        
        # 文本长度检查
        if len(text.strip()) < 10:
            return False
        
        return has_chinese or has_english
    
    def clean_text(self, text: str) -> str:
        """
        清理文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符（保留中文、英文、数字和基本标点）
        # 使用字符类匹配：中文、英文、数字、标点和空白字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：、""''（）【】《》 \t\n\r]', '', text)
        return text.strip()
    
    def generate_summary(self, text: str, max_length: int = None, min_length: int = 20) -> str:
        """
        生成摘要
        
        Args:
            text: 输入文本
            max_length: 最大生成长度
            min_length: 最小生成长度
            
        Returns:
            生成的摘要
        """
        # 验证输入
        if not self.is_valid_text(text):
            raise ValueError("输入不是有效的文本，请提供包含中文或英文的文本内容")
        
        # 清理文本
        text = self.clean_text(text)
        
        # Tokenization
        max_length = max_length or Config.MAX_TARGET_LENGTH
        inputs = self.tokenizer(
            text,
            max_length=Config.MAX_SOURCE_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # 生成摘要
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                min_length=min_length,
                num_beams=Config.NUM_BEAMS,
                length_penalty=Config.LENGTH_PENALTY,
                no_repeat_ngram_size=Config.NO_REPEAT_NGRAM_SIZE,
                early_stopping=True,
                do_sample=False
            )
        
        # 解码
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summary.strip()
    
    def batch_generate(self, texts: list, max_length: int = None, min_length: int = 20) -> list:
        """
        批量生成摘要
        
        Args:
            texts: 输入文本列表
            max_length: 最大生成长度
            min_length: 最小生成长度
            
        Returns:
            生成的摘要列表
        """
        results = []
        for text in texts:
            try:
                summary = self.generate_summary(text, max_length, min_length)
                results.append(summary)
            except Exception as e:
                results.append(f"错误: {str(e)}")
        
        return results

