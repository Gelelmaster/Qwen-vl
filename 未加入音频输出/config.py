import os

# huggingface镜像配置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 模型配置
QWEN_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

# 语音识别模型配置
ASR_MODEL_DIR = r"D:\Desktop\project\Qwen2-VL-2B-Instruct-GPTQ-Int4\models\SenseVoiceSmall"
VAD_MODEL_DIR = r"D:\Desktop\project\Qwen2-VL-2B-Instruct-GPTQ-Int4\models\speech_fsmn_vad_zh-cn-16k-common-pytorch"

# 视觉关键词
VISION_KEYWORDS = ['看到', '看见', '显示', '图像', '图片', '画面', '视频']

# 退出关键词
EXIT_KEYWORDS = ['quit', 'exit', '退出']

# 设备配置
DEVICE = "cuda:0"

# 摄像头配置
CAMERA_WINDOW_NAME = 'Camera'