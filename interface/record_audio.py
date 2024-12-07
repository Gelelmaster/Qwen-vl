import pyaudio
import numpy as np
import time
import io
import wave
import webrtcvad  # 添加 webrtcvad 模块的导入

# 音频设置
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
CHUNK = 1024
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 3
NO_SPEECH_THRESHOLD = 2

# 全局变量
last_active_time = time.time()
segments_to_save = []
interrupt_flag = False

# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(3)

async def record_audio():
    """异步录制音频并返回音频数据"""
    global last_active_time, segments_to_save

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=AUDIO_CHANNELS,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    audio_buffer = []

    while True:
        data = stream.read(CHUNK)
        audio_buffer.append(data)

        # 检查 audio_buffer 中的音频长度
        if len(audio_buffer) * CHUNK / AUDIO_RATE >= 0.5:
            raw_audio = b''.join(audio_buffer)
            vad_result = check_vad_activity(raw_audio)

            if vad_result:
                print("检测到语音")
                last_active_time = time.time()
                segments_to_save.append(raw_audio)
            # else:
            #     print("等待...")

            audio_buffer = []

        # 检测静默时间
        if time.time() - last_active_time > NO_SPEECH_THRESHOLD:
            if segments_to_save:
                # 将录音数据存储到内存中的 BytesIO 对象
                audio_buffer = io.BytesIO()
                wf = wave.open(audio_buffer, 'wb')
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(AUDIO_RATE)
                wf.writeframes(b''.join(segments_to_save))
                audio_buffer.seek(0)
                segments_to_save.clear()
                last_active_time = time.time()
                return audio_buffer

def check_vad_activity(audio_data):
    num, rate = 0, 0.4
    step = int(AUDIO_RATE * 0.02)
    flag_rate = round(rate * len(audio_data) // step)

    for i in range(0, len(audio_data), step):
        chunk = audio_data[i:i + step]
        if len(chunk) == step:
            if vad.is_speech(chunk, sample_rate=AUDIO_RATE):
                num += 1

    return num > flag_rate
