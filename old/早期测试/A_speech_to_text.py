import pyaudio
import threading
import time
import webrtcvad
import numpy as np
import os
from funasr import AutoModel

# 参数设置
AUDIO_RATE = 16000        # 音频采样率
AUDIO_CHANNELS = 1        # 单声道
CHUNK = 1024              # 音频块大小
VAD_MODE = 3              # VAD 模式 (0-3, 数字越大越敏感)
NO_SPEECH_THRESHOLD = 2   # 无效语音阈值，单位：秒

# 全局变量
last_active_time = time.time()
recording_active = True
segments_to_save = []

# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)

# 音频录制线程
def audio_recorder():
    global recording_active, last_active_time, segments_to_save
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=AUDIO_CHANNELS,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    audio_buffer = []
    print("音频录制已开始")
    
    while recording_active:
        data = stream.read(CHUNK)
        audio_buffer.append(data)
        
        # 每 0.5 秒检测一次 VAD
        if len(audio_buffer) * CHUNK / AUDIO_RATE >= 0.5:
            raw_audio = b''.join(audio_buffer)
            vad_result = check_vad_activity(raw_audio)
            
            if vad_result:
                print("检测到语音")
                last_active_time = time.time()
                segments_to_save.append(raw_audio)
            # else:
            #     print("等待语音输入...")
            
            audio_buffer = []  # 清空缓冲区
        
        # 检查无效语音时间
        if time.time() - last_active_time > NO_SPEECH_THRESHOLD:
            if segments_to_save:
                perform_speech_recognition(segments_to_save)
                segments_to_save.clear()
                last_active_time = time.time()
    
    stream.stop_stream()
    stream.close()
    p.terminate()

# 检测 VAD 活动
def check_vad_activity(audio_data):
    num, rate = 0, 0.4
    step = int(AUDIO_RATE * 0.02)  # 20ms 块大小
    flag_rate = round(rate * len(audio_data) // step)

    for i in range(0, len(audio_data), step):
        chunk = audio_data[i:i + step]
        if len(chunk) == step:
            if vad.is_speech(chunk, sample_rate=AUDIO_RATE):
                num += 1

    return num > flag_rate

# 语音识别
def perform_speech_recognition(audio_segments):
    model_dir = r"D:\Desktop\project\Funasr-Qwen-GPTSovits\SenseVoiceSmall"
    vad_model_dir = os.path.join(r"D:\Desktop\project\Funasr-Qwen-GPTSovits\speech_fsmn_vad_zh-cn-16k-common-pytorch")
    # 初始化模型
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
    text = res[0]['text'].split(">")[-1] # .split(">")[-1]：将提取的文本按 > 字符分割，并取最后一个部分。
    print("funasr识别结果:", text)

# 主函数
if __name__ == "__main__":
    try:
        # 启动音频录制线程
        audio_thread = threading.Thread(target=audio_recorder)
        audio_thread.start()
        
        print("按 Ctrl+C 停止录制")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("录制停止中...")
        recording_active = False
        audio_thread.join()
        print("录制已停止")