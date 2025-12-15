"""
配置文件
"""
import os

class Config:
    # 数据路径
    DATA_DIR = "Pre LCSTS"
    TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
    VALID_FILE = os.path.join(DATA_DIR, "eval.csv")
    TEST_FILE = os.path.join(DATA_DIR, "test.csv")
    
    # 模型配置 - 可用的预训练模型列表（用于比较和微调）
    AVAILABLE_MODELS = {
        "1": {
            "name": "google/mt5-base",
            "description": "mT5-base - 基础模型，平衡性能和速度（推荐）",
            "size": "~2.3GB"
        },
        "2": {
            "name": "facebook/mbart-large-cc25",
            "description": "mBART-large - 多语言BART模型，支持中文",
            "size": "~2.5GB"
        },
        "3": {
            "name": "csebuetnlp/mT5_multilingual_XLSum",
            "description": "mT5 XLSum - 专门用于摘要任务的多语言模型",
            "size": "~2.3GB"
        }
    }
    
    # 默认模型
    MODEL_NAME = "google/mt5-base"  # 默认使用mT5-base模型
    MAX_SOURCE_LENGTH = 512  # 原文最大长度
    MAX_TARGET_LENGTH = 128  # 摘要最大长度
    
    # 训练配置
    OUTPUT_DIR = "checkpoints"
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 1000
    SAVE_STEPS = 5000
    EVAL_STEPS = 1000
    LOGGING_STEPS = 100
    
    # 推理配置
    NUM_BEAMS = 4  # beam search的beam数量
    LENGTH_PENALTY = 0.6  # 长度惩罚
    NO_REPEAT_NGRAM_SIZE = 3  # 避免重复的n-gram大小
    
    # 设备配置 - 自动检测最佳设备
    # 优先级: CUDA > MPS (Apple Silicon) > CPU
    @staticmethod
    def get_device():
        """
        获取最佳可用设备
        优先级: CUDA > MPS (Apple Silicon) > CPU
        """
        import torch
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon (M1/M2/M3/M4/M5)
        else:
            return "cpu"
    
    # 自动检测设备（延迟导入torch）
    try:
        import torch
        if torch.cuda.is_available():
            DEVICE = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            DEVICE = "mps"
        else:
            DEVICE = "cpu"
    except:
        DEVICE = "cpu"  # 如果torch未安装，默认CPU
    
    # 随机种子
    SEED = 42
    
    # 镜像配置（用于加速模型下载）
    # 可以通过环境变量 HF_ENDPOINT 设置，或使用 download_model.py 自动设置
    # 推荐镜像: https://hf-mirror.com
    HF_MIRROR = "https://hf-mirror.com"  # Hugging Face 国内镜像

