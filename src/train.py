import os
from unsloth import FastVisionModel
from datasets import load_dataset
from PIL import Image
from configs.kie_config import KIEConfig
import logging
import json
import pandas as pd
from typing import Dict, List
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from unsloth import is_bf16_supported
import base64
import time
import traceback  # 添加到文件开头的导入部分

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),  # 记录到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

class KIETrainer:
    def __init__(self, config: KIEConfig):
        self.config = config
        # 创建日志目录
        log_dir = os.path.join(self.config.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 添加文件处理器
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'training_{time.strftime("%Y%m%d_%H%M%S")}.log'),
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        self.setup_model()
        self.setup_dataset()
        self.setup_trainer()
    
    def setup_model(self):
        """初始化模型和tokenizer"""
        logger.info("正在初始化模型和tokenizer...")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=self.config.model_name,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
        )
        
        # 配置PEFT模型
        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers=self.config.finetune_vision_layers,     # 不微调视觉层
            finetune_language_layers=self.config.finetune_language_layers,    # 微调语言层
            finetune_attention_modules=self.config.finetune_attention_modules,  # 不微调注意力层
            finetune_mlp_modules=self.config.finetune_mlp_modules,        # 微调MLP层
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            random_state=self.config.seed,
            use_rslora=False,
            loftq_config=None,
        )
        
        # 启用训练模式
        FastVisionModel.for_training(self.model)
        
        logger.info("模型初始化完成")
    
    def load_data(self) -> pd.DataFrame:
        """加载CSV数据"""
        try:
            data_path = os.path.join(self.config.train_data_path, "data.csv")
            df = pd.read_csv(data_path)
            logger.info(f"成功加载数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise

    def group_by_image(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """按图片分组数据"""
        grouped_data = {}
        for _, row in df.iterrows():
            image_name = row['图名']
            if image_name not in grouped_data:
                grouped_data[image_name] = []
            grouped_data[image_name].append({
                '属性名': row['属性名'],
                '正确值': row['正确值']
            })
        return grouped_data

    def get_image_path(self, image_name: str) -> str:
        """获取图片路径，支持多种图片格式和文件名格式"""
        try:
            image_extensions = ['.jpg', '.jpeg', '.png']
            base_path = os.path.join(self.config.train_data_path, "images")
            
            # 检查基础路径是否存在
            if not os.path.exists(base_path):
                raise FileNotFoundError(f"图片目录不存在: {base_path}")
            
            # 如果image_name已经包含扩展名
            if any(image_name.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(base_path, image_name)
                if os.path.exists(image_path):
                    return image_path
                raise FileNotFoundError(f"未找到图片文件: {image_name}")
            
            # 如果image_name不包含扩展名，尝试添加不同的扩展名
            for ext in image_extensions:
                image_path = os.path.join(base_path, image_name + ext)
                if os.path.exists(image_path):
                    return image_path
            
            raise FileNotFoundError(f"未找到图片文件: {image_name}，支持的格式: {', '.join(image_extensions)}")
            
        except Exception as e:
            logger.error(f"获取图片路径失败: {str(e)}")
            logger.error(f"图片名称: {image_name}")
            logger.error(f"基础路径: {base_path}")
            logger.error("错误追踪信息:")
            logger.error(traceback.format_exc())
            raise


    def convert_to_conversation(self, image_name: str) -> Dict:
        """将数据转换为对话格式"""
        try:
            # 获取图片的base64编码
            image_path = self.get_image_path(image_name)
            image = Image.open(image_path)
            image = image.resize((512, 512))  # 将图片统一调整为512x512大小
            
            # 获取该图片的所有属性
            if image_name not in self.grouped_data:
                raise KeyError(f"未找到图片对应的属性数据: {image_name}")
            
            image_attributes = self.grouped_data[image_name]
            if not image_attributes:
                raise ValueError(f"图片没有对应的属性数据: {image_name}")
            
            # 构建属性字典
            attributes_dict = {attr['属性名']: attr['正确值'] for attr in image_attributes}
            
            # 将字典转换为JSON字符串，并添加引号使其成为字符串
            attributes_str = f'"{json.dumps(attributes_dict, ensure_ascii=False, indent=2)}"'
            
            # 构建对话格式
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.config.instruction},
                        {"type": "image", "image": image}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": attributes_str
                        }
                    ]
                }
            ]
            
            return {"messages": conversation}
            
        except Exception as e:
            logger.error(f"转换对话格式失败: {str(e)}")
            logger.error(f"图片名称: {image_name}")
            logger.error("错误追踪信息:")
            logger.error(traceback.format_exc())
            raise

    def setup_dataset(self):
        """设置数据集"""
        logger.info("正在加载数据集...")    
        try:
            # 检查数据目录是否存在
            if not os.path.exists(self.config.train_data_path):
                raise FileNotFoundError(f"训练数据目录不存在: {self.config.train_data_path}")
            
            # 加载CSV数据
            df = self.load_data()
            
            # 检查数据是否为空
            if df.empty:
                raise ValueError("CSV文件为空")
            
            # 检查必要的列是否存在
            required_columns = ['图名', '属性名', '正确值']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV文件缺少必要的列: {', '.join(missing_columns)}")
            
            # 按图片分组数据
            self.grouped_data = self.group_by_image(df)
            
            # 检查是否有数据
            if not self.grouped_data:
                raise ValueError("没有找到有效的图片数据")
            
            # 转换数据集格式
            self.processed_dataset = []
            for image_name in self.grouped_data.keys():
                try:
                    conversation = self.convert_to_conversation(image_name)
                    if conversation:
                        self.processed_dataset.append(conversation)
                except Exception as e:
                    logger.error(f"处理图片 {image_name} 时出错: {str(e)}")
                    continue
            
            if not self.processed_dataset:
                raise ValueError("没有成功处理任何数据")
            
            logger.info(f"数据集加载完成，共 {len(self.processed_dataset)} 条记录")
            
        except Exception as e:
            logger.error(f"数据集加载失败: {str(e)}")
            logger.error("错误追踪信息:")
            logger.error(traceback.format_exc())
            raise
    
    def setup_trainer(self):
        """设置训练器"""
        logger.info("正在设置训练器...")
        try:
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
                train_dataset=self.processed_dataset,
                args=SFTConfig(
                    per_device_train_batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    warmup_steps=self.config.warmup_steps,
                    max_steps=self.config.max_steps,
                    learning_rate=self.config.learning_rate,
                    fp16=not is_bf16_supported(),
                    bf16=is_bf16_supported(),
                    logging_steps=self.config.logging_steps,
                    optim="adamw_8bit",
                    weight_decay=self.config.weight_decay,
                    lr_scheduler_type="linear",
                    seed=self.config.seed,
                    output_dir=self.config.output_dir,
                    report_to="tensorboard",  # 启用tensorboard
                    remove_unused_columns=False,
                    dataset_text_field="messages",
                    dataset_kwargs={"skip_prepare_dataset": True},
                    dataset_num_proc=4,
                    max_seq_length=self.config.max_seq_length,
                )
            )
            logger.info("训练器设置完成")
        except Exception as e:
            logger.error(f"训练器设置失败: {str(e)}")
            raise
    
    def train(self):
        """执行训练"""
        logger.info("开始训练...")
        try:
            self.trainer.train()
            logger.info("训练完成")

            # 保存模型
            logger.info(f"正在保存模型到 {self.config.model_save_path}")
            self.trainer.save_model(self.config.model_save_path)
            logger.info("模型保存完成")
        except Exception as e:
            logger.error(f"训练过程出错: {str(e)}")
            logger.error("错误追踪信息:")
            logger.error(traceback.format_exc())
            raise

def main():
    """主函数"""
    try:
        # 加载配置
        config = KIEConfig()
        
        # 创建训练器实例
        trainer = KIETrainer(config)
        
        # 开始训练
        trainer.train()
        
    except Exception as e:
        logger.error(f"训练过程发生错误: {str(e)}")
        logger.error("错误追踪信息:")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
