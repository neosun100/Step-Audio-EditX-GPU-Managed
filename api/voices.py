"""
Built-in voice presets mapping friendly voice IDs to the demo audios provided
with Step-Audio-EditX. These presets are used when the API receives an OpenAI-
style request that only specifies the `voice` field without custom prompt audio.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass(frozen=True)
class VoicePreset:
    """Metadata for a reusable reference audio clip."""

    id: str
    description: str
    prompt_audio: str
    prompt_text: str
    locale: str
    gender: str

    def to_dict(self) -> Dict:
        item = asdict(self)
        return item


VOICE_LIBRARY: Dict[str, VoicePreset] = {
    "fear_female": VoicePreset(
        id="fear_female",
        description="Chinese female - fearful tone",
        prompt_audio="examples/fear_zh_female_prompt.wav",
        prompt_text="我总觉得，有人在跟着我，我能听到奇怪的脚步声。",
        locale="zh-CN",
        gender="female",
    ),
    "happy_en": VoicePreset(
        id="happy_en",
        description="English female - cheerful tone",
        prompt_audio="examples/en_happy_prompt.wav",
        prompt_text="You know, I just finished that big project and feel so relieved.",
        locale="en-US",
        gender="female",
    ),
    "whisper_cn": VoicePreset(
        id="whisper_cn",
        description="Chinese neutral - whisper style",
        prompt_audio="examples/whisper_prompt.wav",
        prompt_text="比如在工作间隙，做一些简单的伸展运动，放松一下身体。",
        locale="zh-CN",
        gender="female",
    ),
    "story_teller": VoicePreset(
        id="story_teller",
        description="Chinese storyteller - neutral",
        prompt_audio="examples/paralingustic_prompt.wav",
        prompt_text="我觉得这个计划大概是可行的，不过还需要再仔细考虑一下。",
        locale="zh-CN",
        gender="female",
    ),
}

DEFAULT_VOICE_ID = "fear_female"


def list_presets() -> List[Dict]:
    """Return preset metadata as list of dictionaries for API responses."""
    return [preset.to_dict() for preset in VOICE_LIBRARY.values()]
