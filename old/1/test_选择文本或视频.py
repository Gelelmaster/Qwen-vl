import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import os
from typing import Optional, Union

# 配置huggingface镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class QwenVLInference:
    def __init__(self):
        # 加载模型和处理器
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4", 
            torch_dtype="auto", 
            device_map="auto"
        )
        
        # 设置分辨率参数
        min_pixels = 256*28*28
        max_pixels = 512*28*28
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4", 
            min_pixels=min_pixels, 
            max_pixels=max_pixels
        )
        
    def inference(self, prompt: str, image: Optional[Image.Image] = None):
        """
        进行推理
        Args:
            prompt: 文本提示
            image: 可选的PIL图像
        """
        if image:
            # 构建包含图像的消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
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
        
        # 处理图像输入(如果有)
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
        
        # 解码输出
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

    def camera_inference(self, camera_number: int = 0):
        """
        使用摄像头进行实时推理
        Args:
            camera_number: 摄像头编号
        """
        cap = cv2.VideoCapture(camera_number)
        
        if not cap.isOpened():
            print("无法访问摄像头")
            return
            
        captured_frame = None
        print("按'c'键进行推理，按'q'键退出")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头图像")
                break

            cv2.imshow('Camera Preview', frame)
            captured_frame = frame

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and captured_frame is not None:
                # 转换为PIL图像
                pil_image = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
                # 调整图像大小
                pil_image = pil_image.resize((512, 512))
                
                # 进行推理
                result = self.inference("请描述这张图片。", pil_image)
                print("推理结果:", result)

        cap.release()
        cv2.destroyAllWindows()

def main():
    # 初始化推理器
    inferencer = QwenVLInference()
    
    while True:
        print("\n请选择模式:")
        print("1. 文本对话")
        print("2. 摄像头图像识别")
        print("3. 退出")
        
        choice = input("请输入选择(1-3): ")
        
        if choice == "1":
            prompt = input("请输入文本: ")
            response = inferencer.inference(prompt)
            print("模型回复:", response)
            
        elif choice == "2":
            print("请选择摄像头:")
            print("0.电脑自带摄像头")
            print("1.外接摄像头")
            camera_num = int(input("请输入摄像头编号: "))
            inferencer.camera_inference(camera_num)
            
        elif choice == "3":
            print("程序退出")
            break
            
        else:
            print("无效选择，请重试")

if __name__ == "__main__":
    main()