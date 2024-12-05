import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import threading
from queue import Queue
import asyncio
import numpy as np
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from A_record_audio import record_audio

class QwenVLInference:
    def __init__(self):
        # 配置huggingface镜像
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        print("正在加载模型...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4", 
            torch_dtype="auto", 
            device_map="auto"
        )
        
        min_pixels = 256*28*28
        max_pixels = 512*28*28
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4", 
            min_pixels=min_pixels, 
            max_pixels=max_pixels
        )
        
        # 用于存储当前帧的队列
        self.frame_queue = Queue(maxsize=1)
        self.running = True
        print("模型加载完成!")

    def camera_thread(self, camera_id=0):
        """摄像头捕获线程"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("无法打开摄像头!")
            self.running = False
            return

        print("摄像头已启动,按'q'键退出")
        while self.running:
            ret, frame = cap.read()
            if ret:
                # 显示摄像头画面
                cv2.imshow('Camera', frame)
                # 更新当前帧
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
            else:
                print("无法读取摄像头画面!")
                break

        cap.release()
        cv2.destroyAllWindows()

    def inference(self, prompt: str, use_camera: bool = False):
        """进行推理"""
        # 检查提示词是否需要视觉输入
        vision_keywords = ['看到', '看见', '显示', '图像', '图片', '画面', '视频']
        need_vision = any(keyword in prompt for keyword in vision_keywords)
        
        if need_vision and use_camera:
            # 获取当前帧
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                # 转换为PIL图像
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # 调整图像大小
                pil_image = pil_image.resize((512, 512))
                
                # 构建包含图像的消息
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            else:
                print("未能获取摄像头画面,仅进行文本对话")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
        else:
            # 仅文本消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

        # 处理输入
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # 推理
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

class AudioState:
    def __init__(self):
        self.current_sound = None
        self.sound_lock = threading.Lock()

async def transcribe_audio(audio_buffer, model):
    """异步使用 FunASR 模型从音频数据流中提取文本"""
    try:
        # 如果是bytes类型，转换为BytesIO对象
        if isinstance(audio_buffer, bytes):
            import io
            audio_buffer = io.BytesIO(audio_buffer)
            
        audio_data = np.frombuffer(audio_buffer.getvalue(), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max  # 归一化处理
        res = model.generate(
            input=audio_data,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        return text
    except Exception as e:
        print(f"语音识别出错: {e}")
        return None

async def audio_main(model, inferencer):
    """主函数，控制音频录制和转录的流程"""
    try:    
        while True:
            print()
            print("***************** Funasr *****************")

            audio_buffer = await record_audio()  # 录制音频
            if audio_buffer is not None:
                text = await transcribe_audio(audio_buffer, model)  # 识别音频
                print("识别结果:", text)  # 输出识别结果
                if text:
                    response = inferencer.inference(text, use_camera=True)
                    print("助手:", response)
            else:
                print("没有检测到有效声音输入，重试...")

    except KeyboardInterrupt:
        print("录制停止中...")

def main():
    inferencer = QwenVLInference()
    
    # 启动摄像头线程
    camera_thread = threading.Thread(target=inferencer.camera_thread)
    camera_thread.start()
    
    # 初始化语音识别模型
    current_dir = os.getcwd()
    model_dir = os.path.join(current_dir, "SenseVoiceSmall")
    vad_model_dir = os.path.join(current_dir, "speech_fsmn_vad_zh-cn-16k-common-pytorch")
    model = AutoModel(
        model=model_dir,
        trust_remote_code=False,
        disable_update=True,
        vad_model=vad_model_dir,
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0",
    )
    print(f"语音识别模型已加载: {model}")

    # 启动语音识别
    asyncio.run(audio_main(model, inferencer))

    # 等待摄像头线程结束
    camera_thread.join()
    print("程序已退出")

if __name__ == "__main__":
    main()