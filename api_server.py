import argparse
import asyncio
import base64
import json
import os
import logging
from pathlib import Path
from typing import List

import uvicorn
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from config.edit_config import get_supported_edit_types
from model_loader import ModelSource
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from whisper_wrapper import WhisperWrapper

from api.schemas import (
    ModelsResponse,
    ModelInfo,
    SpeechRequest,
    VoiceInfo,
)
from api.utils import (
    audio_tensor_to_bytes,
    resolve_input_audio,
    resolve_reference_audio,
)
from api.voices import list_presets

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Step-Audio-EditX FastAPI server")
    parser.add_argument("--model-path", type=str, required=True, help="Path that contains Step-Audio-Tokenizer and Step-Audio-EditX")
    parser.add_argument("--model-source", type=str, default="auto", choices=["auto", "local", "modelscope", "huggingface"])
    parser.add_argument("--tokenizer-model-id", type=str, default="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online")
    parser.add_argument("--tts-model-id", type=str, default=None)
    parser.add_argument("--quantization", type=str, choices=["int4", "int8", "awq-4bit"], default=None)
    parser.add_argument("--torch-dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--enable-auto-transcribe", action="store_true", help="Enable Whisper transcription for edit tasks when no audio_text is provided.")
    parser.add_argument("--awq-model-path", type=str, default=None, help="Path to AWQ quantized model directory (defaults to <model-path>/Step-Audio-EditX-AWQ-4bit if present).")
    parser.add_argument("--bnb-model-path", type=str, default=None, help="Path to BitsAndBytes quantized model directory (defaults to <model-path>/Step-Audio-EditX-bnb-4bit if present).")
    parser.add_argument("--api-host", type=str, default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=8000)
    return parser.parse_args()


def build_fastapi_app(
    model_engines: dict[str, StepAudioTTS],
    model_root: Path,
    asset_roots: list[Path],
    whisper_asr: WhisperWrapper | None,
) -> FastAPI:
    app = FastAPI(
        title="Step-Audio-EditX API",
        version="1.0.0",
        description="OpenAI-compatible Text-to-Speech and audio editing API powered by Step-Audio-EditX.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.model_engines = model_engines
    app.state.model_root = str(model_root)
    app.state.asset_roots = [str(path) for path in asset_roots]
    app.state.whisper_asr = whisper_asr
    app.state.generate_lock = asyncio.Lock()

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models():
        return ModelsResponse(data=[ModelInfo(id="step-audio-editx")])

    @app.get("/v1/voices", response_model=List[VoiceInfo])
    async def list_voices():
        return [VoiceInfo(**preset) for preset in list_presets()]

    @app.get("/v1/tags")
    async def list_supported_tags():
        tags = get_supported_edit_types()
        return {"data": tags}

    async def process_request(request: SpeechRequest):
        options = request.step_audio
        model_variant = options.model_variant if options.model_variant else "base"
        app_engine: StepAudioTTS | None = app.state.model_engines.get(model_variant)
        if app_engine is None:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Model variant '{model_variant}' is not available on this server.")
        whisper_asr: WhisperWrapper | None = app.state.whisper_asr
        lock: asyncio.Lock = app.state.generate_lock
        model_root = app.state.model_root
        asset_roots = app.state.asset_roots
        tmp_paths: List[str] = []

        try:
            loop = asyncio.get_running_loop()

            if options.mode == "clone":
                prompt_path, prompt_text, is_temp = resolve_reference_audio(
                    options.prompt_audio_base64,
                    options.prompt_audio_url,
                    request.voice,
                    asset_roots,
                )
                if is_temp:
                    tmp_paths.append(prompt_path)
                prompt_text = options.prompt_text or prompt_text or request.input

                async with lock:
                    audio_tensor, sr = await loop.run_in_executor(
                        None,
                        app_engine.clone,
                        prompt_path,
                        prompt_text,
                        request.input,
                    )
            else:
                input_path, is_temp = resolve_input_audio(
                    options.input_audio_base64,
                    options.input_audio_url,
                )
                if is_temp:
                    tmp_paths.append(input_path)
                audio_text = options.audio_text
                if not audio_text:
                    if whisper_asr is None:
                        raise HTTPException(status_code=400, detail="audio_text is required when Whisper transcription is disabled.")
                    audio_text = await loop.run_in_executor(None, whisper_asr, input_path)

                edit_text = request.input if options.mode == "paralinguistic" else None

                async with lock:
                    audio_tensor, sr = await loop.run_in_executor(
                        None,
                        app_engine.edit,
                        input_path,
                        audio_text,
                        options.mode,
                        options.edit_info,
                        edit_text,
                        options.intensity,
                    )

            audio_bytes, mime = audio_tensor_to_bytes(audio_tensor, sr, request.response_format)

            headers = {}
            if request.metadata:
                headers.update({f"x-metadata-{k}": v for k, v in request.metadata.items()})

            return Response(
                content=audio_bytes,
                media_type=mime,
                headers=headers | {"X-StepAudio-Model": request.model},
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            for path in tmp_paths:
                if os.path.exists(path):
                    os.remove(path)

    @app.post("/v1/audio/speech")
    async def create_speech(request: SpeechRequest):
        return await process_request(request)

    @app.post("/v1/audio/speech/upload")
    async def create_speech_upload(
        payload: str = Form(..., description="JSON payload compatible with /v1/audio/speech"),
        prompt_audio_file: UploadFile | None = File(
            default=None, description="Optional reference audio file (clone mode)"
        ),
        input_audio_file: UploadFile | None = File(
            default=None, description="Optional edit target audio file"
        ),
    ):
        try:
            payload_dict = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc

        if "step_audio" not in payload_dict or not isinstance(payload_dict["step_audio"], dict):
            payload_dict["step_audio"] = {}

        if prompt_audio_file is not None:
            data = await prompt_audio_file.read()
            payload_dict["step_audio"]["prompt_audio_base64"] = base64.b64encode(data).decode("utf-8")

        if input_audio_file is not None:
            data = await input_audio_file.read()
            payload_dict["step_audio"]["input_audio_base64"] = base64.b64encode(data).decode("utf-8")

        request = SpeechRequest(**payload_dict)
        return await process_request(request)

    return app


def main():
    # 🔥 启用 TF32 加速（与 UI 容器一致）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✅ TF32 acceleration enabled", flush=True)
    
    args = parse_args()

    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_mapping[args.torch_dtype]

    source_mapping = {
        "auto": ModelSource.AUTO,
        "local": ModelSource.LOCAL,
        "modelscope": ModelSource.MODELSCOPE,
        "huggingface": ModelSource.HUGGINGFACE,
    }
    model_source = source_mapping[args.model_source]

    base_dir = Path(args.model_path).resolve()
    project_root = Path(__file__).resolve().parent
    tokenizer_path = base_dir / "Step-Audio-Tokenizer"
    tts_path = base_dir / "Step-Audio-EditX"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer directory missing: {tokenizer_path}")
    if not tts_path.exists():
        raise FileNotFoundError(f"TTS directory missing: {tts_path}")

    encoder = StepAudioTokenizer(
        str(tokenizer_path),
        model_source=model_source,
        funasr_model_id=args.tokenizer_model_id,
    )
    model_engines: dict[str, StepAudioTTS] = {}

    model_engines["base"] = StepAudioTTS(
        str(tts_path),
        encoder,
        model_source=model_source,
        tts_model_id=args.tts_model_id,
        quantization_config=args.quantization,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )

    awq_path = Path(args.awq_model_path) if args.awq_model_path else base_dir / "Step-Audio-EditX-AWQ-4bit"
    if awq_path.exists():
        try:
            model_engines["awq"] = StepAudioTTS(
                str(awq_path),
                encoder,
                model_source=model_source,
                tts_model_id=args.tts_model_id,
                quantization_config="awq-4bit",
                torch_dtype=torch_dtype,
                device_map=args.device_map,
            )
            logger.info(f"✓ AWQ quantized model loaded from {awq_path}")
        except Exception as exc:
            logger.error(f"Failed to load AWQ model at {awq_path}: {exc}")
    else:
        logger.warning(f"AWQ model path {awq_path} not found.")

    bnb_path = Path(args.bnb_model_path) if args.bnb_model_path else base_dir / "Step-Audio-EditX-bnb-4bit"
    if bnb_path.exists():
        try:
            model_engines["bnb"] = StepAudioTTS(
                str(bnb_path),
                encoder,
                model_source=model_source,
                tts_model_id=args.tts_model_id,
                quantization_config="int4",  # BitsAndBytes uses int4
                torch_dtype=torch_dtype,
                device_map=args.device_map,
            )
            logger.info(f"✓ BitsAndBytes quantized model loaded from {bnb_path}")
        except Exception as exc:
            logger.error(f"Failed to load BnB model at {bnb_path}: {exc}")
    else:
        logger.warning(f"BnB model path {bnb_path} not found.")

    whisper_asr = WhisperWrapper() if args.enable_auto_transcribe else None

    asset_roots = [project_root, base_dir]
    app = build_fastapi_app(model_engines, base_dir, asset_roots, whisper_asr)
    uvicorn.run(app, host=args.api_host, port=args.api_port)


if __name__ == "__main__":
    main()
