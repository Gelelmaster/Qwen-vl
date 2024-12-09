import threading
import signal
import os

def force_exit():
    """强制退出程序"""
    print("\n退出程序...")
    os._exit(0)

def create_signal_handler(cleanup_callback):
    """创建信号处理器"""
    def signal_handler(signum, frame):
        print("\n接收到退出信号，正在关闭程序...")
        # 设置定时器，如果正常退出失败则强制退出
        timer = threading.Timer(3.0, force_exit)
        timer.start()
        cleanup_callback()
        raise KeyboardInterrupt
    return signal_handler

from pygame import mixer
from models.gptsovits_model import tts_thread

class TextInputManager:
    def __init__(self, qwen_model, camera_manager, character=None, emotion=None):
        self.qwen_model = qwen_model
        self.camera_manager = camera_manager
        self.running = True
        self.character = character
        self.emotion = emotion

    def text_input_thread(self):
        """文本输入线程"""
        print("\n可以开始文本输入（输入'quit'退出）:")
        while self.running:
            try:
                print("\n文本输入> ", end='', flush=True)
                text = input().strip()
                
                if not self.running:
                    break
                    
                if text.lower() == 'quit':
                    print("\n接收到退出信号，正在关闭程序...")
                    self.running = False
                    break
                    
                if text:
                    # 检查 camera_manager 是否为 None
                    if self.camera_manager is not None:
                        frame = self.camera_manager.get_current_frame()
                    else:
                        frame = None  # 或者使用一个默认值
                        print("未使用摄像头，跳过帧处理\n")
                    
                    response = self.qwen_model.inference(text, frame)
                    print("助手:", response)
                    
                    # 在获取到模型回复后，打断当前的语音输出
                    tts_thread.interrupt_flag = True
                    mixer.stop()  # 停止当前播放
                    # 清空队列
                    with tts_thread.queue.mutex:
                        tts_thread.queue.queue.clear()
                    
                    # 添加新的语音输出
                    # print(f"添加响应到TTS队列: {response}")
                    tts_thread.add_text(response, self.character, self.emotion)
                    # print("响应已添加到TTS队列")
                    
            except Exception as e:
                print(f"处理文本输入时发生错误: {e}")
                continue
