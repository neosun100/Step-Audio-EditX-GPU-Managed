#!/usr/bin/env python3
"""
统一应用 - 同时提供 Gradio UI 和 FastAPI，共享同一个模型实例
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response

# 导入原有的app.py功能
from app import create_interface, load_models

# 导入API功能
from api_server import build_fastapi_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Step-Audio-EditX 统一服务器")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-source", type=str, default="local", choices=["auto", "local", "modelscope", "huggingface"])
    parser.add_argument("--tokenizer-model-id", type=str, default="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online")
    parser.add_argument("--tts-model-id", type=str, default=None)
    parser.add_argument("--quantization", type=str, choices=["int4", "int8", "awq-4bit"], default=None)
    parser.add_argument("--torch-dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--enable-auto-transcribe", action="store_true")
    parser.add_argument("--awq-model-path", type=str, default=None)
    parser.add_argument("--bnb-model-path", type=str, default=None)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("Step-Audio-EditX 统一服务器启动")
    logger.info("=" * 80)
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"服务端口: {args.server_port}")
    logger.info(f"UI + API 共享模型实例")
    logger.info("=" * 80)
    
    # 加载共享模型
    logger.info("正在加载共享模型...")
    encoder, tts_engines, whisper_asr = load_models(args)
    logger.info("✓ 共享模型加载完成")
    
    # 创建 Gradio UI
    logger.info("创建 Gradio UI...")
    demo = create_interface(args, encoder, tts_engines, whisper_asr)
    
    # 创建 FastAPI
    logger.info("创建 FastAPI...")
    model_path = Path(args.model_path)
    asset_roots = [model_path.parent / "examples"] if (model_path.parent / "examples").exists() else []
    
    api_app = build_fastapi_app(
        model_engines=tts_engines,
        model_root=model_path,
        asset_roots=asset_roots,
        whisper_asr=whisper_asr
    )
    
    # 挂载 Gradio 到 FastAPI
    logger.info("挂载 Gradio UI 到 FastAPI...")
    app = gr.mount_gradio_app(api_app, demo, path="/")
    
    logger.info("=" * 80)
    logger.info(f"✓ 服务器启动成功")
    logger.info(f"UI 界面: http://{args.server_name}:{args.server_port}")
    logger.info(f"API 文档: http://{args.server_name}:{args.server_port}/docs")
    logger.info(f"健康检查: http://{args.server_name}:{args.server_port}/healthz")
    logger.info("=" * 80)
    
    # 启动服务器
    uvicorn.run(
        app,
        host=args.server_name,
        port=args.server_port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
