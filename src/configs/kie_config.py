from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class KIEConfig:
    # 模型配置
    model_name: str = "/mnt/d/wsl/work/jupyter/model_hub/Qwen2.5-VL-3B-Instruct"  # 基础模型名称
    max_seq_length: int = 1024            # 最大序列长度
    dtype: Optional[str] = None           # 数据类型，None表示自动选择
    load_in_4bit: bool = True             # 是否使用4bit量化
    
    # LoRA配置
    lora_r: int = 16                      # LoRA rank
    lora_alpha: int = 16                  # LoRA alpha
    lora_dropout: float = 0.05            # LoRA dropout

    # 训练配置更新
    warmup_steps: int = 5                 # 预热步数
    max_steps: int = 250                   # 最大训练步数
    seed: int = 3407                      # 随机种子
    
    instruction: str = "抽取图片中的关键信息,并以json的格式输出"
    
    # 模型配置更新
    finetune_vision_layers: bool = False  # 是否微调视觉层
    finetune_language_layers: bool = True # 是否微调语言层
    finetune_attention_modules: bool = False # 是否微调注意力层
    finetune_mlp_modules: bool = True     # 是否微调MLP层
    
    # 训练配置
    learning_rate: float = 2e-5           # 学习率
    batch_size: int = 2                  # 批次大小
    num_epochs: int = 3                   # 训练轮数
    gradient_accumulation_steps: int = 4   # 梯度累积步数
    warmup_ratio: float = 0.1            # 预热比例
    weight_decay: float = 0.01           # 权重衰减
    
    # 路径配置
    train_data_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/train")   # 训练数据路径          
    eval_data_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/eval")    # 评估数据路径
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "save/kie_output") # 输出目录
    model_save_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "save/kie_final") # 模型保存路径
    model_merge_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "save/kie_merge") # 模型合并路径
    
    # 训练策略
    fp16: bool = True                     # 是否使用混合精度训练
    optim: str = "adamw_torch"            # 优化器选择
    
    # 日志配置
    logging_steps: int = 10               # 日志记录间隔
    save_steps: int = 100                 # 模型保存间隔
    
    # 数据预处理配置
    max_length: int = 512                # 预处理最大长度
    padding: bool = True
    truncation: bool = True


    # 可视化配置
    log_metrics: bool = True              # 是否记录训练指标
    log_model_info: bool = True           # 是否记录模型信息
    tensorboard_dir: str = "tensorboard"  # tensorboard日志目录 
    
    def __post_init__(self):
        """配置验证和初始化"""
        
        # 确保输出目录存在
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        
        assert os.path.exists(self.train_data_path), f"训练数据路径不存在: {self.train_data_path}"
        assert os.path.exists(self.eval_data_path), f"评估数据路径不存在: {self.eval_data_path}"
    


# 创建默认配置实例
default_config = KIEConfig() 
print(default_config)