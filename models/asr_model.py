from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import numpy as np
from config import *

class ASRModel:
    def __init__(self):
        print("正在加载语音识别模型...")
        self.model = AutoModel(
            model=ASR_MODEL_DIR,
            trust_remote_code=False,
            disable_update=True,
            vad_model=VAD_MODEL_DIR,
            vad_kwargs={"max_single_segment_time": 30000},
            device=DEVICE,
        )
        print("语音识别模型加载完成!")

    async def transcribe(self, audio_buffer):
        """从音频数据流中提取文本"""
        try:
            print("开始处理音频数据...") # 调试
            
            if isinstance(audio_buffer, bytes):
                import io
                audio_buffer = io.BytesIO(audio_buffer)
                print("转换音频数据为 BytesIO 对象") # 调试
                
            audio_data = np.frombuffer(audio_buffer.getvalue(), dtype=np.int16)
            print(f"音频数据形状: {audio_data.shape}") # 调试
            
            audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
            print("音频数据归一化完成") # 调试
            
            print("开始语音识别...") # 调试
            res = self.model.generate(
                input=audio_data,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            print(f"语音识别结果: {res}") # 调试
            
            return rich_transcription_postprocess(res[0]["text"])
        except Exception as e:
            
            print(f"语音识别出错: {e}") # 调试
            print(f"错误类型: {type(e)}")

            import traceback # 调试
            print(f"错误堆栈: {traceback.format_exc()}") # 调试
            return None