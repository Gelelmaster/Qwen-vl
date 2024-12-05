from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import sys
from TTS_gptsovits_voice import text_to_speech, get_characters_and_emotions
from TTS_run_model import get_response
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 创建一个信号量来控制并发
semaphore = asyncio.Semaphore(1)
# 创建线程池
executor = ThreadPoolExecutor(max_workers=1)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    user_input: str
    character: str = "default"
    emotion: str = "default"

@app.get("/characters")
async def get_available_characters():
    """获取可用的角色和情感"""
    async with semaphore:
        try:
            logger.info("正在获取角色列表...")
            characters = get_characters_and_emotions()
            logger.info(f"成功获取角色列表: {characters}")
            return characters
        except Exception as e:
            logger.error(f"获取角色列表时出错: {str(e)}", exc_info=True)
            return {"default": ["default"]}

@app.post("/process")
async def process_text(request: ProcessRequest):
    """处理文本请求"""
    async with semaphore:
        try:
            logger.info(f"收到请求: {request}")
            
            # 获取AI响应
            logger.info("正在生成AI响应...")
            response = await get_response(request.user_input)
            logger.info(f"AI响应生成成功: {response}")
            
            # 使用TTS模块进行语音播放
            logger.info("正在进行语音合成...")
            await text_to_speech(response, request.character, request.emotion)
            logger.info("语音合成完成")
            
            return {"response": response}
        except Exception as e:
            logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
            return {"error": str(e)}

@app.on_event("startup")
async def startup_event():
    logger.info("服务器启动...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("服务器关闭...")
    executor.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        workers=1,  # 只使用1个worker
        timeout_keep_alive=300,
        log_level="info"
    )