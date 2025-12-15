"""
数据处理模块
"""
import pandas as pd
import re
from typing import List, Tuple
from transformers import AutoTokenizer
from config import Config


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, model_name: str = None):
        """
        初始化数据处理器
        
        Args:
            model_name: 模型名称，用于加载tokenizer
        """
        model_name = model_name or Config.MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def load_data(self, file_path: str, max_samples: int = None) -> List[Tuple[str, str]]:
        """
        加载CSV数据文件
        
        Args:
            file_path: 数据文件路径
            max_samples: 最大样本数（用于快速测试）
            
        Returns:
            List of (summary, text) tuples
        """
        print(f"正在加载数据: {file_path}")
        data = []
        
        try:
            # 使用chunksize分块读取大文件
            chunk_size = 10000
            chunks = pd.read_csv(file_path, header=None, chunksize=chunk_size, 
                                names=['summary', 'text'], encoding='utf-8')
            
            for chunk in chunks:
                for _, row in chunk.iterrows():
                    summary = str(row['summary']).strip()
                    text = str(row['text']).strip()
                    
                    # 过滤空数据
                    if summary and text and summary != 'nan' and text != 'nan':
                        data.append((summary, text))
                        
                        if max_samples and len(data) >= max_samples:
                            break
                
                if max_samples and len(data) >= max_samples:
                    break
                    
        except Exception as e:
            print(f"加载数据时出错: {e}")
            # 尝试使用更简单的方法
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        summary, text = parts[0].strip(), parts[1].strip()
                        if summary and text:
                            data.append((summary, text))
        
        print(f"成功加载 {len(data)} 条数据")
        return data
    
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
    
    def preprocess_function(self, examples: dict) -> dict:
        """
        预处理函数，用于tokenization
        
        Args:
            examples: 包含'text'和'summary'的字典
            
        Returns:
            处理后的字典
        """
        # 清理文本
        inputs = [self.clean_text(text) for text in examples['text']]
        targets = [self.clean_text(summary) for summary in examples['summary']]
        
        # Tokenization
        model_inputs = self.tokenizer(
            inputs,
            max_length=Config.MAX_SOURCE_LENGTH,
            padding='max_length',
            truncation=True
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=Config.MAX_TARGET_LENGTH,
                padding='max_length',
                truncation=True
            )
        
        # 将padding token的label设为-100（在计算loss时会被忽略）
        labels_input_ids = labels["input_ids"].copy()
        for i in range(len(labels_input_ids)):
            labels_input_ids[i] = [
                -100 if token_id == self.tokenizer.pad_token_id else token_id
                for token_id in labels_input_ids[i]
            ]
        
        model_inputs["labels"] = labels_input_ids
        
        return model_inputs
    
    def prepare_dataset(self, data: List[Tuple[str, str]]) -> dict:
        """
        准备数据集格式
        
        Args:
            data: List of (summary, text) tuples
            
        Returns:
            包含'text'和'summary'的字典
        """
        texts = [item[1] for item in data]
        summaries = [item[0] for item in data]
        
        return {
            'text': texts,
            'summary': summaries
        }

