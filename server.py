#!/usr/bin/env python3
"""
统一服务器 - 同时提供 UI 和 API，共享模型实例
使用懒加载管理器自动管理模型生命周期
"""
import argparse
import logging
import sys
from pathlib import Path

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI
from gradio.routes import mount_gradio_app

from lazy_model_manager import LazyModelManager
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from model_loader import ModelSource
from whisper_wrapper import WhisperWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Step-Audio-EditX 统一服务器")
    parser.add_argument("--model-path", type=str, required=True, help="模型根目录")
    parser.add_argument("--model-source", type=str, default="local", choices=["auto", "local", "modelscope", "huggingface"])
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--enable-auto-transcribe", action="store_true")
    parser.add_argument("--idle-timeout", type=int, default=300, help="模型空闲超时（秒）")
    parser.add_argument("--disable-auto-unload", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860, help="服务端口（UI和API共用）")
    return parser.parse_args()


# 全局模型管理器
global_managers = {}


def init_models(args):
    """初始化模型管理器"""
    logger.info("初始化模型管理器...")
    
    model_path = Path(args.model_path)
    
    # 1. 初始化 Tokenizer（共享）
    tokenizer_path = model_path / "Step-Audio-Tokenizer"
    logger.info(f"加载 Tokenizer: {tokenizer_path}")
    tokenizer = StepAudioTokenizer(
        encoder_path=str(tokenizer_path),
        model_source=args.model_source
    )
    
    # 2. 创建模型工厂函数
    def create_model_factory(path, quant=None):
        def factory():
            torch_dtype = getattr(torch, args.torch_dtype)
            return StepAudioTTS(
                model_path=str(path),
                audio_tokenizer=tokenizer,
                model_source=args.model_source,
                quantization_config=quant,
                torch_dtype=torch_dtype,
                device_map=args.device_map
            )
        return factory
    
    # 3. 注册模型变体
    variants = {
        "base": (model_path / "Step-Audio-EditX", None),
        "awq": (model_path / "Step-Audio-EditX-AWQ-4bit", "awq-4bit"),
        "bnb": (model_path / "Step-Audio-EditX-bnb-4bit", "int4"),
    }
    
    managers = {}
    for variant, (path, quant) in variants.items():
        if path.exists():
            logger.info(f"注册 {variant} 模型: {path}")
            managers[variant] = LazyModelManager(
                model_factory=create_model_factory(path, quant),
                idle_timeout=args.idle_timeout,
                auto_unload=not args.disable_auto_unload
            )
    
    # 4. Whisper ASR
    whisper_asr = None
    if args.enable_auto_transcribe:
        logger.info("初始化 Whisper ASR...")
        whisper_asr = WhisperWrapper()
    
    logger.info(f"模型管理器初始化完成，已注册 {len(managers)} 个变体")
    
    return {
        "tokenizer": tokenizer,
        "managers": managers,
        "whisper_asr": whisper_asr,
        "args": args
    }


def create_gradio_ui(global_state):
    """创建 Gradio UI"""
    managers = global_state["managers"]
    tokenizer = global_state["tokenizer"]
    whisper_asr = global_state["whisper_asr"]
    
    with gr.Blocks(title="Step-Audio-EditX") as demo:
        gr.Markdown("# Step-Audio-EditX - 音频编辑与语音克隆")
        
        with gr.Tab("语音克隆"):
            with gr.Row():
                with gr.Column():
                    prompt_audio = gr.Audio(label="参考音频", type="filepath")
                    prompt_text = gr.Textbox(label="参考文本", lines=2)
                    target_text = gr.Textbox(label="目标文本", lines=2)
                    model_variant = gr.Dropdown(
                        choices=list(managers.keys()),
                        value="base",
                        label="模型变体"
                    )
                    intensity = gr.Slider(0.1, 3.0, value=1.0, label="强度")
                    clone_btn = gr.Button("生成", variant="primary")
                
                with gr.Column():
                    output_audio = gr.Audio(label="输出音频")
                    status_text = gr.Textbox(label="状态", lines=3)
        
        with gr.Tab("模型状态"):
            status_json = gr.JSON(label="模型状态")
            refresh_btn = gr.Button("刷新状态")
            
            def get_status():
                return {v: m.get_status() for v, m in managers.items()}
            
            refresh_btn.click(get_status, outputs=status_json)
            demo.load(get_status, outputs=status_json)
        
        def clone_voice(audio, text, target, variant, intens):
            try:
                if not audio or not text or not target:
                    return None, "错误：请填写所有字段"
                
                # 获取模型（懒加载）
                model = managers[variant].get_model()
                
                # 执行克隆
                result = model.clone(
                    prompt_audio=audio,
                    prompt_text=text,
                    target_text=target,
                    intensity=intens
                )
                
                return result, f"成功！使用模型: {variant}"
            except Exception as e:
                logger.error(f"克隆失败: {e}", exc_info=True)
                return None, f"错误: {str(e)}"
        
        clone_btn.click(
            clone_voice,
            inputs=[prompt_audio, prompt_text, target_text, model_variant, intensity],
            outputs=[output_audio, status_text]
        )
    
    return demo


def create_fastapi_app(global_state):
    """创建 FastAPI"""
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(
        title="Step-Audio-EditX API",
        version="2.0.0",
        description="统一的 TTS 和音频编辑 API"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.state.global_state = global_state
    
    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}
    
    @app.get("/api/v1/models/status")
    async def models_status():
        managers = global_state["managers"]
        return {v: m.get_status() for v, m in managers.items()}
    
    @app.post("/api/v1/models/{variant}/unload")
    async def unload_model(variant: str):
        managers = global_state["managers"]
        if variant not in managers:
            return {"error": f"模型变体 '{variant}' 不存在"}
        managers[variant].force_unload()
        return {"status": "unloaded", "variant": variant}
    
    return app


def main():
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("Step-Audio-EditX 统一服务器")
    logger.info("=" * 80)
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"懒加载: 启用 (空闲超时: {args.idle_timeout}秒)")
    logger.info(f"自动卸载: {'禁用' if args.disable_auto_unload else '启用'}")
    logger.info(f"服务端口: {args.port}")
    logger.info("=" * 80)
    
    # 初始化模型
    global_state = init_models(args)
    
    # 创建 FastAPI
    app = create_fastapi_app(global_state)
    
    # 创建 Gradio UI
    ui = create_gradio_ui(global_state)
    
    # 挂载 Gradio 到 FastAPI
    app = mount_gradio_app(app, ui, path="/")
    
    logger.info("=" * 80)
    logger.info(f"服务器启动: http://{args.host}:{args.port}")
    logger.info(f"UI 界面: http://{args.host}:{args.port}")
    logger.info(f"API 文档: http://{args.host}:{args.port}/docs")
    logger.info(f"健康检查: http://{args.host}:{args.port}/healthz")
    logger.info(f"模型状态: http://{args.host}:{args.port}/api/v1/models/status")
    logger.info("=" * 80)
    
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
    finally:
        for variant, mgr in global_state["managers"].items():
            mgr.shutdown()


if __name__ == "__main__":
    main()
