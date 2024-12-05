import os
import numpy as np
from funasr import AutoModel
from A_audio_vad import audio_recorder

# 语音识别
def perform_speech_recognition(audio_segments):
    model_dir = r"D:\Desktop\project\Funasr-Qwen-GPTSovits\SenseVoiceSmall"
    vad_model_dir = os.path.join(r"D:\Desktop\project\Funasr-Qwen-GPTSovits\speech_fsmn_vad_zh-cn-16k-common-pytorch")
    
    model_senceVoice = AutoModel(
        model=model_dir,
        trust_remote_code=False,
        disable_update=True,
        vad_model=vad_model_dir,
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0",
    )
    
    # 将音频片段合并为一个 numpy 数组
    audio_data = np.frombuffer(b''.join(audio_segments), dtype=np.int16).astype(np.float32)
    
    # 进行语音识别
    res = model_senceVoice.generate(
        input=audio_data,
        cache={},
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False, # 即时文本规范化（ITN）。即时文本规范化通常用于将识别结果转换为更自然的书写格式，例如将数字转换为文字。
        # batch_size_s=60, # 这个参数表示批处理的大小为60秒。这意味着在处理音频时，模型会将音频分成每段60秒的批次进行处理。
        # merge_vad=True, # 表示启用语音活动检测（VAD）合并功能。这个功能会将相邻的语音片段合并在一起，以减少不必要的分段。
        # merge_length_s=15, # 表示合并后的语音片段的最大长度为15秒。超过这个长度的片段将不会被进一步合并。
    ) 
    text = res[0]['text'].split(">")[-1]
    print("funasr识别结果:", text)

# 主函数
if __name__ == "__main__":
    try:
        for segments in audio_recorder():
            perform_speech_recognition(segments)
    
    except KeyboardInterrupt:
        print("录制停止中...")