import threading
import signal
import os
import asyncio
from utils.edgetts_utils import EdgeTTS

def force_exit():
    """强制退出程序"""
    print("\n强制退出程序...")
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

class TextInputManager:
    def __init__(self, qwen_model, camera_manager):
        self.qwen_model = qwen_model
        self.camera_manager = camera_manager
        self.running = True
        self.edge_tts = EdgeTTS()
        
    async def process_text(self, text):
        """处理文本输入的异步方法"""
        frame = self.camera_manager.get_current_frame()
        response = self.qwen_model.inference(text, frame)
        print("助手:", response)
        
        # 添加TTS调用，不指定voice_name让其自动检测语言
        print("开始调用TTS...")
        try:
            # 不指定 voice_name，让系统自动检测语言
            await self.edge_tts.text_to_speech(response)
            print("TTS调用完成")
        except Exception as e:
            print(f"TTS调用失败: {e}")

    def text_input_thread(self):
        """文本输入线程"""
        print("\n可以开始文本输入（输入'quit'退出）:")
        
        # 获取新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.running:
            try:
                text = input("\n文本输入> ")
                if text.lower() == 'quit':
                    self.running = False
                    break
                
                # 在事件循环中运行异步处理
                loop.run_until_complete(self.process_text(text))
                
            except Exception as e:
                print(f"处理文本输入时发生错误: {e}")
                continue
        
        # 关闭事件循环
        loop.close()