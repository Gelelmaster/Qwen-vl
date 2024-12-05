from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import cv2
from config import *

class QwenVLModel:
    def __init__(self):
        print("正在加载QWen模型...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_NAME, 
            torch_dtype="auto", 
            device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained(
            QWEN_MODEL_NAME, 
            min_pixels=MIN_PIXELS, 
            max_pixels=MAX_PIXELS
        )
        print("QWen模型加载完成!")

    def inference(self, prompt: str, frame=None):
        """进行推理"""
        need_vision = any(keyword in prompt for keyword in VISION_KEYWORDS)
        
        if need_vision and frame is not None:
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