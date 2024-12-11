import edge_tts
import asyncio
import pygame
import io
from langdetect import detect
import re

class EdgeTTS:
    def __init__(self):
        try:
            # 初始化pygame mixer用于音频播放
            pygame.mixer.init(frequency=44100)  # 设置更常用的采样率
            pygame.mixer.music.set_volume(1.0)  # 设置最大音量
            print("pygame mixer 初始化成功")
        except Exception as e:
            print(f"pygame mixer 初始化失败: {e}")
        
        # 支持的语音角色示例
        self.voice_list = {
            "中文女声": "zh-CN-XiaoyiNeural",
            "中文男声": "zh-CN-YunjianNeural",
            "英文女声": "en-US-AnaNeural",
            "日语女声": "ja-JP-NanamiNeural",
            "德语女声": "de-DE-KatjaNeural",
            "法语女声": "fr-FR-DeniseNeural",
            "西班牙女声": "ca-ES-JoanaNeural"
        }
        
        # 语言代码到语音的映射
        self.lang_to_voice = {
            'zh': "中文女声",    # 中文
            'en': "英文女声",    # 英语
            'ja': "日语女声",    # 日语
            'de': "德语女声",    # 德语
            'fr': "法语女声",    # 法语
            'ca': "西班牙女声",  # 加泰罗尼亚语/西班牙语
            'es': "西班牙女声"   # 西班牙语
        }

    def detect_language(self, text):
        """检测文本语言并返回对应的语音"""
        try:
            print(f"正在检测语言: {text}")
            
            # 检测是否包含日文字符（平假名、片假名）
            if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
                print("检测到日语")
                return self.lang_to_voice['ja']
            
            # 检测是否包含中文字符
            if re.search(r'[\u4e00-\u9fff]', text):
                print("检测到中文")
                return self.lang_to_voice['zh']
            
            # 使用 langdetect 进行其他语言检测
            lang = detect(text)
            print(f"langdetect 检测结果: {lang}")
            return self.lang_to_voice.get(lang, "英文女声")  # 默认使用英文女声
            
        except Exception as e:
            print(f"语言检测失败: {e}")
            return "中文女声"  # 检测失败时默认使用中文女声

    async def _generate_speech(self, text, voice):
        """生成语音数据"""
        try:
            print(f"开始生成语音数据: {text}")
            communicate = edge_tts.Communicate(text, voice)
            audio_stream = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_stream.write(chunk["data"])
            audio_stream.seek(0)
            print("语音数据生成完成")
            return audio_stream.getvalue()
        except Exception as e:
            print(f"生成语音数据失败: {e}")
            raise

    async def text_to_speech(self, text, voice_name=None):
        """文本转语音主函数"""
        try:
            print(f"\n开始处理文本转语音: {text}")
            
            # 如果没有指定voice_name，自动检测语言
            if voice_name is None:
                voice_name = self.detect_language(text)
                print(f"检测到语言，使��语音: {voice_name}")
            
            # 获取对应的voice
            voice = self.voice_list.get(voice_name, self.voice_list["中文女声"])
            print(f"使用语音角色: {voice}")
            
            # 生成语音数据
            audio_data = await self._generate_speech(text, voice)
            
            # 从内存加载音频
            print("正在加载音频...")
            audio_buffer = io.BytesIO(audio_data)
            pygame.mixer.music.load(audio_buffer)
            
            # 播放音频
            print("开始播放音频...")
            pygame.mixer.music.play()
            
            # 等待播放完成
            print("等待播放完成...")
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
                
            print("播放完成")
            return True
            
        except Exception as e:
            print(f"转换失败: {e}")
            return False
        
        finally:
            try:
                pygame.mixer.music.unload()
            except:
                pass

# 使用示例
if __name__ == "__main__":
    # 创建TTS实例
    tts = EdgeTTS()
    
    # 测试不同语言的文本
    texts = [
        "你好，我是中文",
        "Hello, this is English",
        "こんにちは、日本語です",
        "Bonjour, c'est le français",
        "Hallo, das ist Deutsch",
        "Hola, esto es español"
    ]
    
    # 逐个测试不同语音
    async def test():
        for text in texts:
            print(f"\n正在测试文本: {text}")
            # 不指定voice_name，让系统自动检测
            await tts.text_to_speech(text)
            # 添加一个短暂的延迟，确保音频播放完全结束
            await asyncio.sleep(1)
    
    # 运行测试
    print("开始测试...")
    asyncio.run(test())
    print("测试结束")