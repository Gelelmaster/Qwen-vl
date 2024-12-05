import pyaudio
import time
import webrtcvad

# 参数设置
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
CHUNK = 1024
VAD_MODE = 3
NO_SPEECH_THRESHOLD = 2

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
        
        if len(audio_buffer) * CHUNK / AUDIO_RATE >= 0.5:
            raw_audio = b''.join(audio_buffer)
            vad_result = check_vad_activity(raw_audio)
            
            if vad_result:
                print("检测到语音")
                last_active_time = time.time()
                segments_to_save.append(raw_audio)
            else:
                print("静音中...")
            
            audio_buffer = []

        if time.time() - last_active_time > NO_SPEECH_THRESHOLD:
            if segments_to_save:
                yield segments_to_save  # 返回音频数据流
                segments_to_save.clear()
                last_active_time = time.time()
    
    stream.stop_stream()
    stream.close()
    p.terminate()

# 检测 VAD 活动
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
