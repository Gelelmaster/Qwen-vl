import asyncio
from utils.record_audio import record_audio
from utils.edgetts_utils import EdgeTTS

class AudioManager:
    def __init__(self, qwen_model, camera_manager):
        self.qwen_model = qwen_model
        self.camera_manager = camera_manager
        self.running = True
        self.edge_tts = EdgeTTS()

    async def audio_main(self, asr_model):
        """音频处理主循环"""
        try:    
            while self.running:
                try:
                    audio_buffer = await record_audio()
                    if not self.running:
                        break
                        
                    if audio_buffer is not None:
                        text = await asr_model.transcribe(audio_buffer)
                        if text:
                            print("\n语音输入:", text)
                            frame = self.camera_manager.get_current_frame()
                            response = self.qwen_model.inference(text, frame)
                            print("助手:", response)
                            print("\n文本输入> ", end='', flush=True)
                            
                            # 添加调试信息
                            print("开始调用TTS...")
                            try:
                                await self.edge_tts.text_to_speech(response)
                                print("TTS调用完成")
                            except Exception as e:
                                print(f"TTS调用失败: {e}")
                                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"处理音频时发生错误: {e}")
                    continue

        except Exception as e:
            print(f"音频主循环发生错误: {e}")