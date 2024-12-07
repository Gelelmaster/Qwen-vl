import asyncio
import aiohttp
import logging
from TTS_Funasr import transcribe_audio
from TTS_record_audio import record_audio

# 初始化日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

async def get_characters(session, server_url):
    """从服务器获取可用的角色和情感"""
    for attempt in range(3):  # 添加重试机制
        try:
            async with session.get(f"{server_url}/characters") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"获取到的角色和情感: {data}")
                    return data
                logger.error(f"获取角色列表失败: 状态码 {response.status}")
        except Exception as e:
            logger.error(f"获取角色列表时出错: {str(e)}")
        await asyncio.sleep(2)  # 等待后重试
    
    return {"default": ["default"]}

async def send_to_server(session, server_url, user_input, character, emotion, max_retries=3):
    """向服务器发送请求，带重试机制"""
    timeout = aiohttp.ClientTimeout(total=300)
    
    for attempt in range(max_retries):
        try:
            async with session.post(
                f"{server_url}/process", 
                json={
                    "user_input": user_input,
                    "character": character,
                    "emotion": emotion
                },
                timeout=timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if "response" in data:
                        return data
                    elif "error" in data:
                        logger.error(f"服务器处理错误: {data['error']}")
                else:
                    logger.error(f"服务器响应错误: {response.status}")
                    await asyncio.sleep(5)  # 增加等待时间
        except Exception as e:
            logger.error(f"发送请求时出错 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
        
        if attempt < max_retries - 1:
            await asyncio.sleep(5)  # 增加重试等待时间
    
    return None

async def input_loop(server_url, character, emotion):
    """持续接收用户输入并返回AI的响应"""
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        logger.info("开始语音输入，请说话，输入'退出'结束程序：")
        while True:
            try:
                audio_buffer = await record_audio()
                if audio_buffer is not None:
                    user_input = await transcribe_audio(audio_buffer)
                    logger.info(f"用户输入: {user_input}")

                    if user_input.lower() == '退出。':
                        logger.info("退出程序。")
                        break
                    
                    logger.info("\n正在等待服务器响应...")
                    response = await send_to_server(session, server_url, user_input, character, emotion)
                    
                    if response:
                        logger.info(f"AI回复: {response['response']}")
                    else:
                        logger.error("未能获取服务器响应，请重试")
                else:
                    logger.info("没有检测到有效声音输入，重试...")
            except Exception as e:
                logger.error(f"处理输入时出错: {str(e)}")
                await asyncio.sleep(1)  # 错误后等待1秒再继续

async def main():
    """主函数"""
    server_url = "http://localhost:8080"
    
    async with aiohttp.ClientSession() as session:
        # 从服务器获取角色和情感选项
        characters_and_emotions_dict = await get_characters(session, server_url)
        
        # 列出角色
        character_names = list(characters_and_emotions_dict.keys())
        logger.info(f"\n可用角色：{character_names}")
        
        # 用户选择角色
        character = input("选择角色（按回车键选择默认角色）：")
        if character not in character_names:
            character = character_names[0] if character_names else ""
        
        # 用户选择情感
        emotion_options = characters_and_emotions_dict.get(character, ["default"])
        logger.info(f"\n{character} 可用情感：{emotion_options}")
        
        emotion = input("选择情感（按回车键选择默认情感）：")
        if emotion not in emotion_options:
            emotion = "default"
            
        # 启动输入循环
        await input_loop(server_url, character, emotion)

if __name__ == "__main__":
    asyncio.run(main())