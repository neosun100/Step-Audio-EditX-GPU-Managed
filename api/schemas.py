from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class StepAudioOptions(BaseModel):
    """Step-Audio specific settings embedded inside the OpenAI style payload."""

    mode: Literal["clone", "emotion", "style", "paralinguistic", "speed", "denoise", "vad"] = "clone"
    model_variant: Literal["base", "awq", "bnb"] = "bnb"
    prompt_text: Optional[str] = None
    prompt_audio_base64: Optional[str] = Field(
        default=None,
        description="Reference audio as base64 encoded string. Required for clone if no voice preset is used.",
    )
    prompt_audio_url: Optional[HttpUrl] = Field(
        default=None, description="Optional HTTP(S) url pointing to reference audio file."
    )
    input_audio_base64: Optional[str] = Field(
        default=None,
        description="Existing audio to be edited (base64). Required for edit modes when url is not provided.",
    )
    input_audio_url: Optional[HttpUrl] = Field(
        default=None,
        description="Existing audio to be edited (url). Alternative to base64.",
    )
    audio_text: Optional[str] = Field(
        default=None,
        description="Transcript of the input audio. Required for edit tasks unless Whisper auto-transcription is enabled.",
    )
    edit_info: Optional[str] = Field(
        default=None,
        description="Additional tag such as 'happy', 'shy', 'remove', etc. Depends on edit mode.",
    )
    intensity: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Intensity multiplier (0.1~3.0) describing how subtle or strong the requested edit should be. 0.1=weakest, 1.0=standard, 3.0=strongest.",
    )
    n_edit_iter: int = Field(
        default=1,
        ge=1,
        le=4,
        description="How many iterative edits should be performed (only used for certain edit tasks).",
    )


class SpeechRequest(BaseModel):
    """OpenAI compatible /v1/audio/speech payload with Step-Audio extension."""

    model: str = Field(default="step-audio-editx", description="Model identifier.")
    voice: Optional[str] = Field(
        default=None,
        description="Voice preset alias. If omitted you must provide prompt_audio through step_audio options.",
    )
    input: str = Field(..., description="Target text for synthesis / editing.")
    response_format: Literal["wav", "flac", "mp3"] = Field(default="wav")
    stream: bool = Field(
        default=False,
        description="Reserved. Streaming is not yet supported via the REST API.",
    )
    step_audio: StepAudioOptions = Field(
        default_factory=StepAudioOptions,
        description="Step-Audio specific settings.",
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional metadata bag forwarded in the response for bookkeeping.",
    )


class ModelInfo(BaseModel):
    """OpenAI style /v1/models entry."""

    id: str
    object: Literal["model"] = "model"
    owned_by: str = "stepfun-ai"


class ModelsResponse(BaseModel):
    data: list[ModelInfo]


class VoiceInfo(BaseModel):
    id: str
    description: str
    prompt_audio: str
    prompt_text: str
    locale: str
    gender: str
