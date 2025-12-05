from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Tuple

import requests
import soundfile as sf
from pydub import AudioSegment

from api.voices import VOICE_LIBRARY, VoicePreset, DEFAULT_VOICE_ID


def _to_str(url: Optional[str]) -> Optional[str]:
    if url is None:
        return None
    return str(url)


def _download_temp_file(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    suffix = Path(url).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fp:
        fp.write(response.content)
        return fp.name


def _write_base64_audio(data: str, suffix: str = ".wav") -> str:
    raw = base64.b64decode(data)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fp:
        fp.write(raw)
        return fp.name


def resolve_reference_audio(
    prompt_audio_base64: Optional[str],
    prompt_audio_url: Optional[str],
    voice_id: Optional[str],
    search_roots: Iterable[str],
) -> Tuple[str, str, bool]:
    """
    Determine which reference audio/path to use for cloning.
    Returns tuple of (audio_path, prompt_text).
    """

    tmp_path = None

    if prompt_audio_base64:
        tmp_path = _write_base64_audio(prompt_audio_base64)
        return tmp_path, "", True

    prompt_audio_url = _to_str(prompt_audio_url)
    if prompt_audio_url:
        tmp_path = _download_temp_file(prompt_audio_url)
        return tmp_path, "", True

    preset_id = voice_id or DEFAULT_VOICE_ID

    if preset_id not in VOICE_LIBRARY:
        raise ValueError(f"Unknown voice preset '{preset_id}'. Provide prompt_audio or use one of: {', '.join(VOICE_LIBRARY.keys())}")

    preset: VoicePreset = VOICE_LIBRARY[preset_id]
    for root in search_roots:
        preset_path = Path(root) / preset.prompt_audio
        if preset_path.exists():
            return str(preset_path), preset.prompt_text, False
    raise FileNotFoundError(
        f"Preset audio '{preset.prompt_audio}' not found in any of: {', '.join(map(str, search_roots))}"
    )


def resolve_input_audio(
    audio_base64: Optional[str],
    audio_url: Optional[str],
) -> Tuple[str, bool]:
    """Return local path to audio clip that should be edited."""

    if audio_base64:
        return _write_base64_audio(audio_base64), True
    audio_url = _to_str(audio_url)
    if audio_url:
        return _download_temp_file(audio_url), True
    raise ValueError("An existing audio clip is required for this mode. Provide step_audio.input_audio_base64 or step_audio.input_audio_url.")


def audio_tensor_to_bytes(
    waveform,
    sample_rate: int,
    response_format: str,
) -> Tuple[bytes, str]:
    """
    Convert torch Tensor -> audio bytes in the desired format.
    Returns bytes and corresponding MIME type.
    """

    data = waveform.squeeze().cpu().numpy()
    buffer = io.BytesIO()
    fmt = response_format.lower()

    if fmt == "wav":
        sf.write(buffer, data, sample_rate, format="WAV")
        mime = "audio/wav"
    elif fmt == "flac":
        sf.write(buffer, data, sample_rate, format="FLAC")
        mime = "audio/flac"
    elif fmt == "mp3":
        tmp = io.BytesIO()
        sf.write(tmp, data, sample_rate, format="WAV")
        tmp.seek(0)
        audio_seg = AudioSegment.from_file(tmp, format="wav")
        audio_seg.export(buffer, format="mp3")
        mime = "audio/mpeg"
    else:
        raise ValueError(f"Unsupported response_format '{response_format}'. Choose from wav, flac, mp3.")

    buffer.seek(0)
    return buffer.read(), mime
