import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
from PIL import Image
import numpy as np

# 下载模型
model_dir = snapshot_download("qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4")

# 加载模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)

# 加载处理器
processor = AutoProcessor.from_pretrained(model_dir)

# 获取摄像头, 摄像头的设备编号通常从 0 开始，改为 1 使用第二个摄像头
print("请使用键盘上的数字键选择要使用的摄像头：0.电脑自带摄像头，1.手机前置摄像头，2.手机后置摄像头")
camera_number = int(input())
cap = cv2.VideoCapture(camera_number)

# 检查摄像头是否正常工作
if not cap.isOpened():
    print("无法访问摄像头")
    exit()

# 使用一个标志来表示是否捕获了一帧图像
captured_frame = None

# 打开摄像头窗口并实时显示图像
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头图像")
        break

    # 显示摄像头窗口
    cv2.imshow('Captured Image', frame)

    # 保存捕获的最后一帧图像
    captured_frame = frame

    # 等待按键退出窗口
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按下 'q' 键退出摄像头预览
        break
    if key == ord('c'):  # 按下 'c' 键进行推理
        if captured_frame is not None:
            # 将 OpenCV 图像 (numpy.ndarray) 转换为 PIL 图像
            pil_image = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))

            # 调整图像尺寸（例如，将图像调整为适合模型的大小）
            pil_image_resized = pil_image.resize((512, 512))  # 调整到 512x512 大小，根据需要修改

            # 传递给 messages，使用 PIL 图像对象
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image_resized},  # 使用 PIL 图像对象
                        {"type": "text", "text": "请描述这张图片。"},
                    ],
                }
            ]

            # 为推理准备数据
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 传递图像对象
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # 推理：生成输出
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # 打印生成的文本
            print(output_text)

# 释放摄像头
cap.release()

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
