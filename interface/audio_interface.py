import asyncio
from interface.record_audio import record_audio

class AudioManager:
    def __init__(self, qwen_model, camera_manager):
        self.qwen_model = qwen_model
        self.camera_manager = camera_manager
        self.running = True

    async def audio_main(self, asr_model):
        """音频处理主循环"""
        print("开始音频处理主循环")  # 调试
        try:    
            while self.running:
                try:
                    print("准备录音...")  # 调试
                    audio_buffer = await record_audio()
                    print(f"录音完成，数据大小: {len(audio_buffer) if audio_buffer is not None else 'None'}")  # 调试

                    if not self.running:
                        break
                        
                    if audio_buffer is not None:
                        print("开始语音识别...")  # 调试
                        text = await asr_model.transcribe(audio_buffer)
                        print(f"语音识别结果: {text if text else 'None'}")  # 调试
                        if text:
                            print("\n语音输入:", text)
                            frame = self.camera_manager.get_current_frame()
                            response = self.qwen_model.inference(text, frame)
                            print("助手:", response)
                            print("\n文本输入> ", end='', flush=True)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"处理音频时发生错误: {e}")
                    continue

        except Exception as e:
            print(f"音频主循环发生错误: {e}")