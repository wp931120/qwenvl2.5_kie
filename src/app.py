import os
import json
import gradio as gr
import torch
import logging
import base64
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoProcessor
from configs.kie_config import KIEConfig
# 清理GPU缓存
torch.cuda.empty_cache()

# 配置logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载配置
config = KIEConfig()

###############################################################################
# 初始化模型和处理器
###############################################################################
try:
    logger.info(f"正在加载模型: {config.model_name}")

    # 初始化vllm模型
    llm = LLM(
        model=config.model_name,  # 使用基础模型路径
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,  # 降低内存使用率
        dtype=torch.bfloat16,
        trust_remote_code=True,
        max_num_seqs=5,      # 允许批处理
        mm_processor_kwargs={
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        max_model_len=2*config.max_seq_length,  # 限制模型长度
        enable_lora=True,
        max_lora_rank=32
    )
  
    # Also create the processor for building prompts + handling images
    processor = AutoProcessor.from_pretrained(config.model_name)
    logger.info("模型加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}", exc_info=True)
    raise

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.1,  # 降低温度以获得更确定性的输出
    top_p=0.9,
    max_tokens=2048,  # 增加最大生成长度
    stop=None 
)
lora_request = LoRARequest(lora_name = "vision", lora_int_id = 1, lora_local_path = config.model_save_path)

# 添加全局变量用于控制任务中断
is_processing = False
cancel_signal = False

def process_image(image):
    """处理上传的图片并进行预测"""
    global is_processing, cancel_signal
    
    # 如果已经在处理中，则返回
    if is_processing:
        return "任务正在处理中，请等待..."
    
    is_processing = True
    cancel_signal = False
    image_data = None
    
    try:
        # 检查图像是否为None
        if image is None:
            is_processing = False
            return "请先上传图片再进行分析"
        
        # 获取图像路径
        if isinstance(image, str):
            image_path = image
        elif hasattr(image, 'name'):
            image_path = image.name
        else:
            # 如果image既不是字符串也没有name属性，可能是PIL图像对象
            image_data = image if isinstance(image, Image.Image) else Image.fromarray(image)
            # 创建临时文件路径
            temp_path = os.path.join(os.path.dirname(__file__), "temp_image.jpg")
            image_data.save(temp_path)
            image_path = temp_path
            image_data = image_data
        
        # 如果是路径，打开图像
        if isinstance(image_path, str):
            try:
                image_data = Image.open(image_path)
                #image_data = image_data.resize((512, 512))  # 将图片统一调整为512x512大小
            except Exception as e:
                logger.error(f"打开图像失败: {str(e)}")
                return f"打开图像失败: {str(e)}"
        
        # 构建Qwen2.5-VL格式的消息
        qwen_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": config.instruction},
                    {"type": "image", "image": f"{image_path}"}
              
                ]
            }
        ]
        
        # 使用处理器构建提示
        prompt = processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 准备多模态数据
        mm_data = {"image": [image_data]}
        
        # 构建LLM输入
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        
        # 运行推理
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params,lora_request=lora_request)
        
        if not outputs or not outputs[0].outputs:
            is_processing = False
            return "模型未返回有效输出"
            
        response = outputs[0].outputs[0].text
        is_processing = False
        return response
            
    except Exception as e:
        logger.error(f"处理失败: {str(e)}", exc_info=True)
        is_processing = False
        return f"处理失败: {str(e)}"

def cancel_process():
    """取消当前处理任务"""
    global cancel_signal
    if is_processing:
        cancel_signal = True
        return "已发送取消信号，正在停止处理..."
    else:
        return "当前没有正在进行的任务"

def clear_output():
    """清空输出结果"""
    return None, None


# 创建Gradio界面
with gr.Blocks(title="智能图像信息抽取系统", theme=gr.themes.Soft(primary_hue="slate", secondary_hue="teal")) as demo:
    # 更新顶部样式，添加动画效果
    gr.HTML("""
        <div style="text-align: center; margin: 2rem auto; max-width: 1200px;">
            <div class="header-container" style="background: linear-gradient(135deg, #e2e8f0, #cbd5e1, #94a3b8); 
                        padding: 2.5rem; 
                        border-radius: 30px; 
                        box-shadow: 0 20px 40px rgba(148, 163, 184, 0.15);
                        transform: translateY(0);
                        transition: all 0.3s ease;">
                <div style="display: flex; 
                           align-items: center; 
                           justify-content: center; 
                           gap: 2rem;
                           margin-bottom: 1rem;">
                    <div class="icon-wrapper" style="animation: float 3s ease-in-out infinite;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" 
                             stroke="#475569" stroke-width="2" style="filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.08));">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                            <line x1="16" y1="13" x2="8" y2="13"></line>
                            <line x1="16" y1="17" x2="8" y2="17"></line>
                        </svg>
                    </div>
                    <h1 class="title-glow" style="font-weight: 700; 
                               font-size: 2.6rem; 
                               color: #475569; 
                               text-shadow: 1px 1px 2px rgba(0,0,0,0.08);
                               margin: 0;
                               padding: 0;">智能图像信息抽取系统</h1>
                    <div class="icon-wrapper" style="animation: float 3s ease-in-out infinite;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" 
                             stroke="#475569" stroke-width="2" style="filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.08));">
                            <circle cx="12" cy="12" r="10"></circle>
                            <path d="M12 16c2.2 0 4-1.8 4-4s-1.8-4-4-4-4 1.8-4 4 1.8 4 4 4z"></path>
                        </svg>
                    </div>
                </div>
                <p class="subtitle-fade" style="margin: 0.8rem 0 0; 
                          color: #64748b; 
                          font-size: 1.2rem; 
                          font-weight: 500;
                          max-width: 800px;
                          text-align: center;">
                    基于先进视觉语言模型，提供高效精准的图像信息抽取服务 
                    <span class="animate-pulse">🚀</span>
                </p>
            </div>
        </div>
    """)

    with gr.Column(elem_id="container"):
        with gr.Row(equal_height=True, variant="panel"):
            # 左侧列
            with gr.Column(scale=1, min_width=700):
                gr.HTML("""
                    <div class="section-title">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="3" width="18" height="18" rx="2"></rect>
                            <circle cx="8.5" cy="8.5" r="1.5"></circle>
                            <path d="M20.4 14.5L16 10 4 20"></path>
                        </svg>
                        <span>图像输入区域</span>
                    </div>
                """)
                
                # 调整图片显示区域
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="",
                        type="filepath",
                        elem_id="input-image",
                        height=600,
                        width=600,
                        container=True,
                        show_label=False,
                    )
                
                # 按钮区域垂直对齐
                with gr.Row(equal_height=True, elem_id="button-group"):
                    process_btn = gr.Button("🚀 开始智能分析", variant="primary", elem_id="process-btn", size="lg", scale=2)
                    cancel_btn = gr.Button("⛔ 中断处理", variant="stop", elem_id="cancel-btn", size="lg")
                    clear_btn = gr.Button("🧹 清空结果", variant="secondary", elem_id="clear-btn", size="lg")
            
            # 右侧列
            with gr.Column(scale=1, min_width=700):
                gr.HTML("""
                    <div class="section-title">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                            <line x1="16" y1="13" x2="8" y2="13"></line>
                            <line x1="16" y1="17" x2="8" y2="17"></line>
                        </svg>
                        <span>分析结果区域</span>
                    </div>
                """)
                
                with gr.Column(scale=1):
                    status_indicator = gr.Textbox(
                        label="",
                        elem_id="status-indicator",
                        value="就绪",
                        interactive=False
                    )
                    
                    output_text = gr.Textbox(
                        label="",
                        elem_id="output-text",
                        lines=35,
                        max_lines=200,
                        show_copy_button=True,
                        interactive=False,
                        container=True,
                        scale=2
                    )

    # 更新CSS样式
    gr.HTML("""
    <style>
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    @keyframes glow {
        0% { text-shadow: 0 0 5px rgba(71, 85, 105, 0.1); }
        50% { text-shadow: 0 0 20px rgba(71, 85, 105, 0.2); }
        100% { text-shadow: 0 0 5px rgba(71, 85, 105, 0.1); }
    }

    .header-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(148, 163, 184, 0.2);
    }

    .title-glow {
        animation: glow 3s ease-in-out infinite;
    }

    .subtitle-fade {
        opacity: 0;
        animation: fadeIn 1s ease-out forwards;
    }

    .animate-pulse {
        animation: pulse 2s infinite;
    }

    #container {
        max-width: 1600px;
        margin: 0 auto;
        background: linear-gradient(135deg, #f8fafc, #f0f9ff);
        padding: 2rem;
        border-radius: 24px;
        box-shadow: 0 12px 24px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }

    #container:hover {
        box-shadow: 0 16px 32px rgba(0,0,0,0.08);
    }

    .section-title {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(to right, #94a3b8, #cbd5e1);
        border-radius: 12px;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(148, 163, 184, 0.2);
        transition: all 0.3s ease;
    }

    .section-title:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(148, 163, 184, 0.3);
    }

    #input-image {
        border-radius: 16px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        background: white;
        padding: 1.5rem;
        margin: 1rem auto;
        display: flex;
        justify-content: center;
    }

    #input-image:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.08);
    }

    #button-group {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.5rem;
    }

    #process-btn, #cancel-btn, #clear-btn {
        transition: all 0.3s ease;
        transform: translateY(0);
    }

    #process-btn:hover, #cancel-btn:hover, #clear-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }

    #process-btn {
        background: linear-gradient(135deg, #94a3b8, #64748b);
        border: none;
    }

    #cancel-btn {
        background: linear-gradient(135deg, #f87171, #ef4444);
        border: none;
    }

    #clear-btn {
        background: linear-gradient(135deg, #cbd5e1, #94a3b8);
        border: none;
        color: white;
    }

    #status-indicator {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        font-weight: 600;
        text-align: center;
        margin: 1rem auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        font-size: 1.1rem;
        color: #64748b;
        width: 90%;
        transition: all 0.3s ease;
    }

    #status-indicator.processing {
        background: linear-gradient(270deg, #94a3b8, #64748b);
        background-size: 200% 200%;
        animation: gradient 2s ease infinite;
        color: white;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    #output-text {
        min-height: 600px !important;
        border-radius: 16px;
        background: white;
        padding: 1.5rem;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
        margin: 0 auto;
        width: 90%;
        transition: all 0.3s ease;
    }

    #output-text:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.08);
    }

    .dark #container {
        background: linear-gradient(135deg, #1e293b, #0f172a);
    }

    .dark #input-image, .dark #output-text, .dark #status-indicator {
        background: #1e293b;
        color: #e2e8f0;
    }

    .dark .section-title {
        background: linear-gradient(to right, #64748b, #94a3b8);
    }

    @media (max-width: 768px) {
        #container {
            padding: 1rem;
        }
        
        .title-glow {
            font-size: 2rem;
        }
        
        #button-group {
            flex-direction: column;
        }
        
        #input-image, #output-text {
            width: 100%;
        }
    }
    </style>
    """)

    
    # 处理按钮点击事件
    process_btn.click(
        fn=lambda: "处理中...",
        inputs=None,
        outputs=status_indicator
    ).then(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_text]
    ).then(
        fn=lambda: "完成",
        inputs=None,
        outputs=status_indicator
    )
    
    # 取消按钮点击事件
    cancel_btn.click(
        fn=cancel_process,
        inputs=None,
        outputs=status_indicator
    )
    
    # 清空按钮点击事件
    clear_btn.click(
        fn=clear_output,
        inputs=None,
        outputs=[input_image, output_text]
    ).then(
        fn=lambda: "就绪",
        inputs=None,
        outputs=status_indicator
    )

if __name__ == "__main__":
    logger.info("启动服务")
    # 修改启动参数，使用localhost或0.0.0.0，并关闭share
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)