import os
import json
import logging
import traceback
from PIL import Image
from typing import Dict, Optional
from configs.kie_config import KIEConfig
from train import FastVisionModel
import argparse
from configs.kie_config import KIEConfig
import json
import os

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KIEInference:
    def __init__(self, config: KIEConfig):
        self.config = config
        self.setup_model()
    
    def setup_model(self):
        """初始化模型和tokenizer"""
        logger.info("正在初始化模型和tokenizer...")
        try:
            # 加载训练好的模型权重
            if os.path.exists(self.config.model_save_path):
                logger.info(f"成功加载模型权重: {self.config.model_save_path}")
            else:
                raise FileNotFoundError(f"模型权重文件不存在: {self.config.model_save_path}")
                
            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                model_name=self.config.model_save_path,
                dtype=self.config.dtype,
                load_in_4bit=self.config.load_in_4bit,
            )
            FastVisionModel.for_inference(self.model) # Enable for inference!
            logger.info("模型初始化完成")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            logger.error("错误追踪信息:")
            logger.error(traceback.format_exc())
            raise
    
    def get_image_path(self, image_name: str) -> str:
        """获取图片路径，支持多种图片格式和文件名格式"""
        image_extensions = ['.jpg', '.jpeg', '.png']
        base_path = os.path.join(self.config.eval_data_path, "images")
        try: 
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
    
    def predict(self, image_name: str) -> Dict:
        """对单张图片进行预测"""
        try:
            # 获取图片路径并加载图片
            image_path = self.get_image_path(image_name)
            image = Image.open(image_path)
            image = image.resize((512, 512))  # 将图片统一调整为512x512大小

            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": self.config.instruction},
                    {"type": "image", "image": image}
                ]}
            ]
            # 使用模型生成预测
            input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True)
            inputs = self.tokenizer(
                image,
                input_text,
                add_special_tokens = False,
                return_tensors = "pt",
            ).to("cuda")
            output = self.model.generate(**inputs, max_new_tokens = self.config.max_seq_length,
                   use_cache = True, temperature = 1.5, min_p = 0.1)
            
            # 解码模型输出的token IDs为文本
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # 提取模型生成的部分（去除输入提示）
            response = response[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
            print(f"模型输出: {response}")
            
            # 解析预测结果
            try:
                predicted_attributes = json.loads(response.strip('"'))
                return {
                    'image_name': image_name,
                    'predictions': predicted_attributes
                }
            except json.JSONDecodeError:
                logger.error(f"无法解析模型输出为JSON: {response}")
                return None
            
        except Exception as e:
            logger.error(f"预测图片 {image_name} 失败: {str(e)}")
            logger.error("错误追踪信息:")
            logger.error(traceback.format_exc())
            return None
    
    def batch_predict(self, image_names: list[str]) -> list[Optional[Dict]]:
        """批量预测多张图片"""
        results = []
        for image_name in image_names:
            try:
                result = self.predict(image_name)
                results.append(result)
            except Exception as e:
                logger.error(f"处理图片 {image_name} 时出错: {str(e)}")
                results.append(None)
        return results

def main():
    """主函数，用于批量推理评估数据集中的图片"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='KIE推理程序')
    parser.add_argument('--config', type=str, default='configs/kie_config.py',
                       help='配置文件路径')
    args = parser.parse_args()

    try:
        # 加载配置  
        config = KIEConfig()
        # 初始化推理器
        inferencer = KIEInference(config)
        
        # 获取评估数据集中的所有图片
        eval_images_dir = os.path.join(config.eval_data_path, "images")
        image_names = [f for f in os.listdir(eval_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_names:
            logger.error(f"未在{eval_images_dir}目录下找到任何图片")
            return
        
        logger.info(f"找到{len(image_names)}张图片待处理")
        
        # 批量推理
        results = inferencer.batch_predict(["/mnt/d/wsl/work/kie_qwen25vl/src/data/eval/images/27de47bb9d9f4d4a95b55b16b08a7528_5.png"])    
        
        # 保存结果
        output_dir = os.path.join(config.eval_data_path, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "predictions.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"推理结果已保存至: {output_file}")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        logger.error("错误追踪信息:")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
  