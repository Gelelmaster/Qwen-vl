import asyncio
from interface.record_audio import record_audio, interrupt_flag
from models.gptsovits_model import text_to_speech, tts_thread

class AudioManager:
    def __init__(self, qwen_model, camera_manager, character=None, emotion=None):
        self.qwen_model = qwen_model
        self.camera_manager = camera_manager
        self.running = True
        self.character = character
        self.emotion = emotion
        
    async def audio_main(self, asr_model):
        """音频处理主循环"""
        try:
            print("开始音频处理主循环")
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
                            
                            # print(f"添加响应到TTS队列: {response}")
                            tts_thread.add_text(response, self.character, self.emotion)
                            # print("响应已添加到TTS队列")
                                
                            print("\n文本输入> ", end='', flush=True)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"处理音频时发生错误: {e}")
                    import traceback
                    print(traceback.format_exc())
                    continue

        except Exception as e:
            print(f"音频主循环发生错误: {e}")
            import traceback
            print(traceback.format_exc())
