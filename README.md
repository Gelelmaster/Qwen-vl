# Qwen-vl
离线模型，实现语音文本对话，视频识别等功能。

python 3.9 以上，模型是 qwen-2b-vl ，要求显存 8G 以上，cuda 环境。  
  

下载 ASR 模型和 VAD 文件放到 models 目录下
- SenseVoiceSmall
``` sh
git clone https://huggingface.co/FunAudioLLM/SenseVoiceSmall
```
- speech_fsmn_vad_zh-cn-16k-common-pytorch
``` sh
git clone https://www.modelscope.cn/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch.git
```

下载语音模型放到 GPT_SoVITS/pretrained_models 目录下
- [chinese-hubert-base](https://huggingface.co/Gelel/chinese-hubert-base/tree/main)
- [chinese-roberta-wwm-ext-large](https://huggingface.co/Gelel/chinese-roberta-wwm-ext-large/tree/main)

    
可通过 [GPT_SoVITS](https://github.com/RVC-Boss/GPT-SoVITS/tree/main) 训练角色语音，放到 trained 文件夹下  
  
运行程序：
```bash
python main.py
```
