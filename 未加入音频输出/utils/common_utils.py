import threading
import signal
import os

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

    def text_input_thread(self):
        """文本输入线程"""
        print("\n可以开始文本输入（输入'quit'退出）:")
        while self.running:
            try:
                user_input = input("\n文本输入> ").strip()
                if user_input.lower() in ['quit', 'exit', '退出']:
                    self.running = False
                    break
                if user_input:
                    frame = self.camera_manager.get_current_frame()
                    response = self.qwen_model.inference(user_input, frame)
                    print("助手:", response)
            except EOFError:
                self.running = False
                break
            except Exception as e:
                print(f"文本输入处理错误: {e}")
                continue