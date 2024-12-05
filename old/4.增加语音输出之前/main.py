import asyncio
import signal
import threading
from models.qwen_model import QwenVLModel
from models.asr_model import ASRModel
from utils.camera_utils import CameraManager
from utils.audio_utils import AudioManager
from utils.common_utils import TextInputManager, create_signal_handler

class Application:
    def __init__(self):
        self.running = True
        self.camera_manager = CameraManager()
        self.qwen_model = QwenVLModel()
        self.asr_model = ASRModel()
        self.audio_manager = AudioManager(self.qwen_model, self.camera_manager)
        self.text_manager = TextInputManager(self.qwen_model, self.camera_manager)

    def cleanup(self):
        """清理所有资源"""
        print("\n开始清理应用资源...")
        self.running = False
        self.audio_manager.running = False
        self.text_manager.running = False
        self.camera_manager.cleanup()

    async def run(self):
        """运行应用"""
        try:
            # 注册信号处理器
            signal.signal(signal.SIGINT, create_signal_handler(self.cleanup))
            signal.signal(signal.SIGTERM, create_signal_handler(self.cleanup))

            # 启动摄像头线程
            camera_thread = threading.Thread(
                target=self.camera_manager.camera_thread_func
            )
            camera_thread.daemon = True
            camera_thread.start()

            # 启动文本输入线程
            text_thread = threading.Thread(
                target=self.text_manager.text_input_thread
            )
            text_thread.daemon = True
            text_thread.start()

            # 运行音频处理循环
            await self.audio_manager.audio_main(self.asr_model)

        except KeyboardInterrupt:
            print("\n程序正在关闭...")
        except Exception as e:
            print(f"运行时发生错误: {e}")
        finally:
            self.cleanup()
            
            # 等待线程结束
            if camera_thread.is_alive():
                camera_thread.join(timeout=1.0)
            if text_thread.is_alive():
                text_thread.join(timeout=1.0)

def main():
    """主函数"""
    app = Application()
    
    try:
        # 获取事件循环
        loop = asyncio.get_event_loop()
        # 运行应用
        loop.run_until_complete(app.run())
    except Exception as e:
        print(f"主程序发生错误: {e}")
    finally:
        # 关闭事件循环
        loop.close()
        print("程序已退出")

if __name__ == "__main__":
    main()