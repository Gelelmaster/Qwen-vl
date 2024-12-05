import asyncio
import signal
import threading
from models.qwen_model import QwenVLModel
from models.asr_model import ASRModel
from interface.camera_interface import CameraManager
from interface.audio_interface import AudioManager
from interface.common_interface import TextInputManager, create_signal_handler

class Application:
    def __init__(self):
        self.running = True
        self.camera_manager = CameraManager()
        self.qwen_model = QwenVLModel()
        self.asr_model = ASRModel()
        self.audio_manager = AudioManager(self.qwen_model, self.camera_manager)
        self.text_manager = TextInputManager(self.qwen_model, self.camera_manager)

        # 检查可用摄像头
        self.available_cameras = self.camera_manager.list_cameras()
        if self.available_cameras:
            print("检测到以下摄像头:")
            for idx, cam in enumerate(self.available_cameras):
                print(f"{idx}: {cam}")
            selected_camera = int(input("请选择摄像头编号 (或输入-1跳过): "))
            if selected_camera != -1:
                self.camera_manager.select_camera(selected_camera)
            else:
                self.camera_manager = None
        else:
            print("未检测到摄像头，跳过摄像头功能。")
            self.camera_manager = None

    def cleanup(self):
        """清理所有资源"""
        print("\n开始清理应用资源...")
        self.running = False
        self.audio_manager.running = False
        self.text_manager.running = False
        if self.camera_manager:
            self.camera_manager.cleanup()

    async def run(self):
        """运行应用"""
        try:
            # 注册信号处理器
            signal.signal(signal.SIGINT, create_signal_handler(self.cleanup))
            signal.signal(signal.SIGTERM, create_signal_handler(self.cleanup))

            # 启动摄像头线程（如果有选择）
            if self.camera_manager:
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

            # # 初始化角色和情感字典
            # characters_and_emotions_dict = self.tts_model.get_characters_and_emotions()
            
            # # 列出角色
            # character_names = list(characters_and_emotions_dict.keys())
            # print(f"\n可用角色：{character_names}")
            
            # # 用户选择角色
            # character = input("选择角色（按回车键选择默认角色）：")
            # if character not in character_names:
            #     character = character_names[0] if character_names else ""
            
            # # 用户选择情感
            # emotion_options = characters_and_emotions_dict.get(character, ["default"])
            # print(f"\n{character} 可用情感：{emotion_options}")
            
            # emotion = input("选择情感（按回车键选择默认情感）：")
            # if emotion not in emotion_options:
            #     emotion = "default"

            # 运行音频处理循环
            await self.audio_manager.audio_main(self.asr_model)

            # # 使用TTS模块进行语音播放
            # print("\n生成的模型回复将以语音输出...")
            # await self.tts_model.text_to_speech("欢迎使用应用程序", character, emotion)

        except KeyboardInterrupt:
            print("\n程序正在关闭...")
        except Exception as e:
            print(f"运行时发生错误: {e}")
        finally:
            self.cleanup()
            
            # 等待线程结束
            if self.camera_manager and camera_thread.is_alive():
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