# Qwen-vl
离线模型，实现语音文本对话，视频识别等功能

#### 下载 ASR 模型和 VAD 文件放到 models 目录下
- [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/tree/main)
``` sh
git clone https://huggingface.co/FunAudioLLM/SenseVoiceSmall
```
- [speech_fsmn_vad_zh-cn-16k-common-pytorch](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/files)  
``` sh
git clone https://www.modelscope.cn/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch.git

**中国地区的用户可以[在此处下载这些模型](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/dkxgpiy9zb96hob4#nVNhX)。**

1. 从 [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) 下载预训练模型，并将其放置在 `GPT_SoVITS/pretrained_models` 目录中。

2. 从 [G2PWModel_1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip) 下载模型，解压并重命名为 `G2PWModel`，然后将其放置在 `GPT_SoVITS/text` 目录中。（仅限中文TTS）

3. 对于 UVR5（人声/伴奏分离和混响移除，额外功能），从 [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) 下载模型，并将其放置在 `tools/uvr5/uvr5_weights` 目录中。

4. 对于中文 ASR（额外功能），从 [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files)、[Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files) 和 [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) 下载模型，并将它们放置在 `tools/asr/models` 目录中。

5. 对于英语或日语 ASR（额外功能），从 [Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3) 下载模型，并将其放置在 `tools/asr/models` 目录中。此外，[其他模型](https://huggingface.co/Systran) 可能具有类似效果且占用更少的磁盘空间。
