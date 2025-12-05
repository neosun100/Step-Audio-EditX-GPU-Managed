"""
音频编辑配置模块
包含支持的编辑类型和相关配置
"""

def get_supported_edit_types():
    """
    获取支持的编辑类型和选项（中英文对照）

    Returns:
        Dict[str, list]: Dictionary of edit types and their options
    """
    return {
        "clone (克隆)": [],
        "emotion (情感)": [
            'happy (开心)', 'angry (生气)', 'sad (悲伤)', 'humour (幽默)', 
            'confusion (困惑)', 'disgusted (厌恶)', 'empathy (同情)', 
            'embarrass (尴尬)', 'fear (恐惧)', 'surprised (惊讶)', 
            'excited (兴奋)', 'depressed (沮丧)', 'coldness (冷漠)', 
            'admiration (钦佩)', 'remove (移除)'
        ],
        "style (风格)": [
            'serious (严肃)', 'arrogant (傲慢)', 'child (童声)', 'older (老年)', 
            'girl (少女)', 'pure (纯净)', 'sister (御姐)', 'sweet (甜美)', 
            'ethereal (空灵)', 'whisper (耳语)', 'gentle (温柔)', 'recite (朗诵)', 
            'generous (大方)', 'act_coy (撒娇)', 'warm (温暖)', 'shy (害羞)', 
            'comfort (安慰)', 'authority (权威)', 'chat (聊天)', 'radio (播音)', 
            'soulful (深情)', 'story (讲故事)', 'vivid (生动)', 'program (节目)', 
            'news (新闻)', 'advertising (广告)', 'roar (咆哮)', 'murmur (低语)', 
            'shout (喊叫)', 'deeply (深沉)', 'loudly (响亮)', 'remove (移除)', 
            'exaggerated (夸张)'
        ],
        "vad (语音活动检测)": [],
        "denoise (降噪)": [],
        "paralinguistic (副语言)": [],
        "speed (语速)": [
            "faster (更快)", "slower (更慢)", 
            "more faster (非常快)", "more slower (非常慢)"
        ],
    }


def get_edit_type_key(display_name):
    """
    从显示名称中提取实际的编辑类型键
    
    Args:
        display_name: 显示名称，如 "emotion (情感)"
        
    Returns:
        str: 实际的键，如 "emotion"
    """
    if '(' in display_name:
        return display_name.split(' (')[0]
    return display_name


def get_edit_info_key(display_name):
    """
    从显示名称中提取实际的编辑信息键
    
    Args:
        display_name: 显示名称，如 "happy (开心)"
        
    Returns:
        str: 实际的键，如 "happy"
    """
    if '(' in display_name:
        return display_name.split(' (')[0]
    return display_name
