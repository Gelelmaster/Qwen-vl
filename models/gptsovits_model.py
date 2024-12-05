import io
import numpy as np
import soundfile as sf
from pygame import mixer
import torch
import asyncio
import threading
from Synthesizers.base import Base_TTS_Synthesizer
from importlib import import_module
from src.common_config_manager import app_config

class GPTSovitsModel:
    def __init__(self):
        print("正在初始化语音合成模型...")
        # 检查CUDA可用性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'} for TTS inference.")

        # 动态导入语音合成器模块
        synthesizer_name = app_config.synthesizer
        synthesizer_module = import_module(f"Synthesizers.{synthesizer_name}")
        TTS_Synthesizer = synthesizer_module.TTS_Synthesizer

        # 创建合成器实例
        self.tts_synthesizer: Base_TTS_Synthesizer = TTS_Synthesizer(debug_mode=True)
        
        # 初始化音频播放
        mixer.init(frequency=32000, size=-16, channels=2, buffer=256)
        
        # 音频状态管理
        self.sound_lock = threading.Lock()
        self.current_sound = None
        
        self.characters_and_emotions = {}
        print("语音合成模型初始化完成!")

    def get_characters_and_emotions(self):
        """获取角色和情感信息"""
        if not self.characters_and_emotions:
            self.characters_and_emotions = self.tts_synthesizer.get_characters()
        return self.characters_and_emotions

    async def get_audio(self, data):
        """生成音频数据"""
        if not data.get("text"):
            raise ValueError("文本不能为空")

        try:
            task = self.tts_synthesizer.params_parser(data)
            if hasattr(task, 'to'):
                task = task.to(self.device)
            
            gen = self.tts_synthesizer.generate(task, return_type="numpy")
            audio_data = next(gen)
            return audio_data

        except Exception as e:
            print(f"音频生成错误: {e}")
            raise

    async def play_audio(self, audio_data, sample_rate=32000):
        """播放生成的音频数据"""
        if isinstance(audio_data, tuple):
            sample_rate, audio_data = audio_data

        audio_data = np.array(audio_data)
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, sample_rate, format='WAV')
        audio_buffer.seek(0)

        with self.sound_lock:
            if self.current_sound is not None:
                self.current_sound.stop()
            self.current_sound = mixer.Sound(audio_buffer)
            self.current_sound.play()

        while mixer.get_busy():
            await asyncio.sleep(0.1)

    async def text_to_speech(self, text, character="", emotion="default"):
        """文本转语音"""
        try:
            data = {"text": text, "character": character, "emotion": emotion}
            
            with self.sound_lock:
                if self.current_sound is not None:
                    self.current_sound.stop()

            print("正在生成音频...")
            audio_data = await self.get_audio(data)
            print("音频生成完成，开始播放...")
            await self.play_audio(audio_data)
            
        except Exception as e:
            print(f"语音合成错误: {e}")