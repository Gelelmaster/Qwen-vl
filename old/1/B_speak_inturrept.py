import pyaudio
import wave
import threading
import time
import numpy as np

# 初始化pyaudio
p = pyaudio.PyAudio()

# 播放音频的函数
def play_audio(file_path):
    # 打开音频文件
    wf = wave.open(file_path, 'rb')

    # 打开音频流
    stream = p.open(format=pyaudio.paInt16,
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    frames_per_buffer=1024)

    # 读取并播放音频文件
    data = wf.readframes(1024)
    while len(data) > 0:
        # 如果检测到打断标志，则停止播放
        if interrupt_flag:
            print("用户打断，停止播放音频")
            stream.stop_stream()
            break
        stream.write(data)
        data = wf.readframes(1024)

    # 停止播放流
    stream.stop_stream()
    stream.close()

# 监听用户输入的麦克风音频
def listen_for_input():
    global interrupt_flag
    r = pyaudio.PyAudio()
    mic_stream = r.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024)

    while True:
        # 获取音频数据
        audio_data = np.frombuffer(mic_stream.read(1024), dtype=np.int16)
        # 如果检测到用户有声音，则认为是打断
        if np.abs(audio_data).max() > 500:  # 阈值可调整
            interrupt_flag = True
            break

# 主函数
def main():
    global interrupt_flag
    interrupt_flag = False

    # 设置播放的音频文件路径
    audio_file_path = r"D:\Desktop\project\GPT-Sovits-Infer\tmp_audio\派蒙\开心_happy\【开心_happy】嗯…有了，我们把刚刚拿到的团子牛奶分你一瓶。这样你也算本轮比赛的「胜者」了！.wav"  # 替换为你的音频文件路径

    # 创建并启动监听用户输入的线程
    listen_thread = threading.Thread(target=listen_for_input, daemon=True)
    listen_thread.start()

    # 播放音频的线程
    play_thread = threading.Thread(target=play_audio, args=(audio_file_path,))
    play_thread.start()

    # 等待直到用户打断或播放结束
    play_thread.join()
    print("程序结束")

if __name__ == "__main__":
    main()
