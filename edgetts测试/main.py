import asyncio
import signal
import threading
from models.qwen_model import QwenVLModel
from models.asr_model import ASRModel
from interface.role_select import get_character_list, select_character
from interface.camera_utils import CameraManager
from interface.audio_utils import AudioManager
from interface.common_utils import TextInputManager, create_signal_handler

class Application:
    def __init__(self):
        # 初始化所有属性为 None
        self.running = True
        self.camera_manager = None
        self.qwen_model = None
        self.asr_model = None
        self.audio_manager = None
        self.text_manager = None
        
        try:
            self.camera_manager = CameraManager()
            
            # 如果摄像头不可用，设置 camera_manager 为 None
            if not self.camera_manager.camera_enabled:
                print("\n未发现摄像头，默认不使用")
                self.camera_manager = None
                
            # 获取角色列表并选择角色
            characters = get_character_list()
            character, emotion = select_character(characters)
                
            self.qwen_model = QwenVLModel()
            self.asr_model = ASRModel()
            self.audio_manager = AudioManager(self.qwen_model, self.camera_manager, character=character, emotion=emotion)
            self.text_manager = TextInputManager(self.qwen_model, self.camera_manager, character=character, emotion=emotion)
            
        except KeyboardInterrupt:
            print("\n程序初始化被用户取消")
            self.cleanup()
            self.running = False
        except Exception as e:
            print(f"\n初始化过程发生错误: {e}")
            self.cleanup()
            self.running = False

    def cleanup(self):
        """清理所有资源"""
        print("\n开始清理应用资源...")
        self.running = False
        
        if self.audio_manager is not None:
            self.audio_manager.running = False
        if self.text_manager is not None:
            self.text_manager.running = False
        if self.camera_manager is not None:
            print("开始清理摄像头资源...")
            self.camera_manager.cleanup()
            print("摄像头资源清理完成")

    async def run(self):
        """运行应用"""
        camera_thread = None
        text_thread = None
        
        try:
            # 注册信号处理器
            signal.signal(signal.SIGINT, create_signal_handler(self.cleanup))
            signal.signal(signal.SIGTERM, create_signal_handler(self.cleanup))

            # 只有在camera_manager存在且启用的情况下才启动摄像头线程
            if self.camera_manager is not None and self.camera_manager.camera_enabled:
                camera_thread = threading.Thread(
                    target=self.camera_manager.camera_thread_func
                )
                camera_thread.daemon = True
                camera_thread.start()

            # 只有在text_manager存在的情况下才启动文本输入线程
            if self.text_manager is not None:
                text_thread = threading.Thread(
                    target=self.text_manager.text_input_thread
                )
                text_thread.daemon = True
                text_thread.start()

            # 只有在audio_manager和asr_model都存在的情况下才运行音频处理循环
            if self.audio_manager is not None and self.asr_model is not None:
                await self.audio_manager.audio_main(self.asr_model)

        except KeyboardInterrupt:
            print("\n程序正在关闭...")
        except Exception as e:
            print(f"运行时发生错误: {e}")
        finally:
            self.cleanup()
            
            # 只在线程存在时等待其结束
            if camera_thread and camera_thread.is_alive():
                camera_thread.join(timeout=1.0)
            if text_thread and text_thread.is_alive():
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
