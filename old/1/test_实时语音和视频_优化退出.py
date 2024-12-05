import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import os
import threading
from queue import Queue
import asyncio
import numpy as np
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from A_record_audio import record_audio
import signal
from concurrent.futures import ThreadPoolExecutor

# 配置huggingface镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class QwenVLInference:
    def __init__(self):
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
        
        # 初始化属性
        self.frame_queue = Queue(maxsize=1)
        self.running = True
        self.cap = None
        self.camera_thread = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        print("模型加载完成!")

    def cleanup(self):
        """清理资源"""
        self.running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.executor.shutdown(wait=False)

    def camera_thread_func(self, camera_id=0):
        """摄像头捕获线程"""
        self.cap = cv2.VideoCapture(camera_id)
        try:
            if not self.cap.isOpened():
                print("无法打开摄像头!")
                self.running = False
                return

            print("摄像头已启动,按'q'键退出")
            while self.running:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break
                    
                cv2.imshow('Camera', frame)
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break

        except Exception as e:
            print(f"摄像头线程发生错误: {e}")
        finally:
            self.cleanup()

    def inference(self, prompt: str, use_camera: bool = False):
        """进行推理"""
        vision_keywords = ['看到', '看见', '显示', '图像', '图片', '画面', '视频']
        need_vision = any(keyword in prompt for keyword in vision_keywords)
        
        if need_vision and use_camera:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil_image = pil_image.resize((512, 512))
                
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
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

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
        if isinstance(audio_buffer, bytes):
            import io
            audio_buffer = io.BytesIO(audio_buffer)
            
        audio_data = np.frombuffer(audio_buffer.getvalue(), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
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
    """音频处理主循环"""
    try:    
        while inferencer.running:
            print("\n***************** Funasr *****************")
            try:
                audio_buffer = await record_audio()
                if not inferencer.running:
                    break
                    
                if audio_buffer is not None:
                    text = await transcribe_audio(audio_buffer, model)
                    if text:
                        print("识别结果:", text)
                        response = inferencer.inference(text, use_camera=True)
                        print("助手:", response)
                else:
                    print("没有检测到有效声音输入，重试...")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"处理音频时发生错误: {e}")
                continue

    except Exception as e:
        print(f"音频主循环发生错误: {e}")

def signal_handler(signum, frame):
    """信号处理函数"""
    print("\n接收到退出信号，正在关闭程序...")
    raise KeyboardInterrupt

def main():
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    inferencer = QwenVLInference()
    
    try:
        # 启动摄像头线程
        inferencer.camera_thread = threading.Thread(target=inferencer.camera_thread_func)
        inferencer.camera_thread.daemon = True
        inferencer.camera_thread.start()
        
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

        # 运行音频处理循环
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(audio_main(model, inferencer))
        except KeyboardInterrupt:
            pass
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    except KeyboardInterrupt:
        print("\n程序正在关闭...")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 清理资源
        inferencer.cleanup()
        # 确保程序完全退出
        os._exit(0)

if __name__ == "__main__":
    main()