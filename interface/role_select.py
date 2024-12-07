from Synthesizers.base import Base_TTS_Synthesizer
from importlib import import_module
from src.common_config_manager import app_config

def get_character_list():
    """获取可用角色列表"""
    # 动态导入语音合成器
    synthesizer_name = app_config.synthesizer
    synthesizer_module = import_module(f"Synthesizers.{synthesizer_name}")
    TTS_Synthesizer = synthesizer_module.TTS_Synthesizer
    
    # 创建合成器实例并获取角色列表
    tts_synthesizer = TTS_Synthesizer(debug_mode=True)
    return tts_synthesizer.get_characters()

def select_character(characters):
    """选择角色和情感"""
    # 显示角色列表
    print("\n可用角色列表:")
    char_list = list(characters.keys())
    for i, char in enumerate(char_list):
        emotions = characters[char]
        print(f"{i+1}. {char} (情感: {', '.join(emotions)})")
    
    # 选择角色
    while True:
        choice = input("\n请选择角色编号: ").strip()
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(char_list):
                character = char_list[choice_num - 1]
                break
            else:
                print(f"请输入1到{len(char_list)}之间的数字")
        except ValueError:
            print("请输入数字，而不是文字")
    
    # 选择情感
    emotions = characters[character]
    print(f"\n可用情感: {', '.join(emotions)}")
    while True:
        emotion = input("请选择情感 (默认为 default): ").strip()
        if not emotion:
            emotion = "default"
        if emotion in emotions:
            break
        print(f"无效的情感，可用选项: {', '.join(emotions)}")
    
    return character, emotion