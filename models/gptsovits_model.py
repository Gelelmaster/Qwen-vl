import io
import numpy as np
import torch
import soundfile as sf
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # 隐藏 pygame 的欢迎信息
from pygame import mixer
import asyncio
import threading
import queue
import time
import pyaudio
from Synthesizers.base import Base_TTS_Synthesizer
from importlib import import_module
from src.common_config_manager import app_config

# 初始化设备和合成器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

synthesizer_name = app_config.synthesizer
synthesizer_module = import_module(f"Synthesizers.{synthesizer_name}")
TTS_Synthesizer = synthesizer_module.TTS_Synthesizer
tts_synthesizer = TTS_Synthesizer(debug_mode=False)

# 初始化音频播放
mixer.init(frequency=32000, size=-16, channels=2, buffer=256)

class TTSThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self.running = True
        self.current_task = None
        self.interrupt_flag = False
        # print("TTS线程已初始化")
        
    def run(self):
        # print("TTS线程开始运行")
        
        # 创建监听麦克风的线程
        def listen_for_input():
            try:
                r = pyaudio.PyAudio()
                stream = r.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=16000,
                              input=True,
                              frames_per_buffer=1024)
                
                while self.running:
                    try:
                        data = stream.read(1024, exception_on_overflow=False)
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        # 如果检测到声音，设置中断标志
                        if np.abs(audio_data).max() > 500:  # 阈值可调
                            self.interrupt_flag = True
                            mixer.stop()  # 立即停止当前播放
                            # 清空队列中的所有任务
                            with self.queue.mutex:
                                self.queue.queue.clear()
                            time.sleep(0.1)  # 短暂等待以避免立即重新触发
                    except Exception as e:
                        print(f"监听输入错误: {e}")
                        
            finally:
                stream.stop_stream()
                stream.close()
                r.terminate()
        
        # 启动监听线程
        listen_thread = threading.Thread(target=listen_for_input, daemon=True)
        listen_thread.start()
        
        while self.running:
            try:
                # print("等待新的TTS任务...")
                try:
                    self.current_task = self.queue.get(timeout=1.0)
                    # print(f"获取到新任务: {self.current_task}")
                except queue.Empty:
                    continue
                
                if self.current_task is None:
                    print("收到停止信号")
                    break
                    
                text, character, emotion = self.current_task
                # print(f"开始处理文本: {text}, 角色: {character}, 情感: {emotion}")
                
                # 重置中断标志
                self.interrupt_flag = False
                
                # 生成语音
                data = {"text": text, "character": character, "emotion": emotion}
                # print("正在解析参数...")
                task = tts_synthesizer.params_parser(data)
                if hasattr(task, 'to'):
                    task = task.to(device)
                
                if self.interrupt_flag:  # 检查是否被中断
                    continue
                    
                # print("开始生成语音...")
                gen = tts_synthesizer.generate(task, return_type="numpy")
                audio_data = next(gen)
                print("语音生成完成", end="\n\n") # end="\n\n" 表示换行两次
                
                if self.interrupt_flag:  # 再次检查是否被中断
                    continue
                
                if isinstance(audio_data, tuple):
                    sample_rate, audio_data = audio_data
                
                audio_data = np.array(audio_data)
                if audio_data.ndim == 1:
                    audio_data = audio_data.reshape(-1, 1)
                
                # 播放音频
                # print("准备播放音频...")
                audio_buffer = io.BytesIO()
                sf.write(audio_buffer, audio_data, 32000, format='WAV')
                audio_buffer.seek(0)
                
                if self.interrupt_flag:  # 最后检查一次是否被中断
                    continue
                
                sound = mixer.Sound(audio_buffer)
                # print("开始播放音频...")
                sound.play()
                
                # 等待播放完成或被中断
                while mixer.get_busy() and not self.interrupt_flag:
                    time.sleep(0.1)
                
                if self.interrupt_flag:
                    print("播放被中断")
                    mixer.stop()
                else:
                    # print("音频播放完成")
                    continue
                
            except Exception as e:
                print(f"TTS线程错误: {e}")
                import traceback
                print(traceback.format_exc())
            finally:
                if self.current_task:
                    self.queue.task_done()
                    self.current_task = None
    
    def stop(self):
        """停止线程"""
        # print("正在停止TTS线程...")
        self.running = False
        mixer.stop()
        self.queue.put(None)
        
    def add_text(self, text, character, emotion="default"):
        """添加要转换的文本到队列"""
        # print(f"添加文本到TTS队列: {text}")
        self.queue.put((text, character, emotion))

# 创建全局TTS线程实例
# print("正在创建TTS线程...")
tts_thread = TTSThread()
# print("启动TTS线程...")
tts_thread.start()
# print("TTS线程已启动")

async def text_to_speech(text, character, emotion="default"):
    """异步接口：将文本添加到TTS队列"""
    # print(f"收到TTS请求: {text}")
    tts_thread.add_text(text, character, emotion)
    # print("TTS请求已添加到队列")
