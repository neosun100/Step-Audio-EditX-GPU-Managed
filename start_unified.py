#!/usr/bin/env python3
"""
统一启动脚本 - 同时启动 UI 和 API，共享同一个模型实例
支持懒加载和自动卸载
"""
import argparse
import logging
import os
import sys
import threading
from pathlib import Path

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 项目导入
from lazy_model_manager import LazyModelManager
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from model_loader import ModelSource
from whisper_wrapper import WhisperWrapper

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Step-Audio-EditX 统一服务器")
    
    # 模型配置
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-source", type=str, default="local", choices=["auto", "local", "modelscope", "huggingface"])
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--enable-auto-transcribe", action="store_true")
    
    # 懒加载配置
    parser.add_argument("--idle-timeout", type=int, default=300, help="空闲超时（秒）")
    parser.add_argument("--disable-auto-unload", action="store_true", help="禁用自动卸载")
    
    # 服务器配置
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--ui-port", type=int, default=7860)
    parser.add_argument("--api-port", type=int, default=8000)
    
    return parser.parse_args()


class SharedModelManager:
    """共享模型管理器"""
    
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model_managers = {}
        self.whisper_asr = None
        
        self._init_all()
    
    def _init_all(self):
        """初始化所有组件"""
        logger.info("初始化共享模型管理器...")
        
        # 1. 初始化 Tokenizer（共享，不懒加载）
        model_path = Path(self.args.model_path)
        tokenizer_path = model_path / "Step-Audio-Tokenizer"
        
        logger.info(f"加载 Tokenizer: {tokenizer_path}")
        self.tokenizer = StepAudioTokenizer(
            model_path=str(tokenizer_path),
            model_source=self.args.model_source
        )
        
        # 2. 注册模型变体（懒加载）
        variants = {
            "base": model_path / "Step-Audio-EditX",
            "awq": model_path / "Step-Audio-EditX-AWQ-4bit",
            "bnb": model_path / "Step-Audio-EditX-bnb-4bit",
        }
        
        for variant, path in variants.items():
            if path.exists():
                logger.info(f"注册 {variant} 模型: {path}")
                quant = "awq-4bit" if variant == "awq" else ("int4" if variant == "bnb" else None)
                self.model_managers[variant] = LazyModelManager(
                    model_factory=lambda p=path, q=quant: self._create_model(p, q),
                    idle_timeout=self.args.idle_timeout,
                    auto_unload=not self.args.disable_auto_unload
                )
        
        # 3. 初始化 Whisper
        if self.args.enable_auto_transcribe:
            logger.info("初始化 Whisper ASR...")
            self.whisper_asr = WhisperWrapper()
        
        logger.info(f"共享模型管理器初始化完成，已注册 {len(self.model_managers)} 个模型变体")
    
    def _create_model(self, model_path, quantization):
        """创建模型实例"""
        torch_dtype = getattr(torch, self.args.torch_dtype)
        return StepAudioTTS(
            model_path=str(model_path),
            audio_tokenizer=self.tokenizer,
            model_source=self.args.model_source,
            quantization_config=quantization,
            torch_dtype=torch_dtype,
            device_map=self.args.device_map
        )
    
    def get_model(self, variant="base"):
        """获取模型（懒加载）"""
        if variant not in self.model_managers:
            variant = "base"
        return self.model_managers[variant].get_model()
    
    def get_status(self):
        """获取状态"""
        return {v: m.get_status() for v, m in self.model_managers.items()}


def create_ui(manager: SharedModelManager, args):
    """创建 Gradio UI"""
    import app
    
    # 修改 app.py 使用共享管理器
    # 这里简化处理，直接导入并修改全局变量
    app.shared_manager = manager
    
    # 创建接口
    interface = app.create_interface(args, manager.tokenizer, manager.whisper_asr)
    return interface


def create_api(manager: SharedModelManager):
    """创建 FastAPI"""
    api = FastAPI(title="Step-Audio-EditX API", version="2.0.0")
    
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    api.state.manager = manager
    
    @api.get("/healthz")
    async def healthz():
        return {"status": "ok"}
    
    @api.get("/v1/models/status")
    async def models_status():
        return manager.get_status()
    
    # 导入 API 路由
    import api_server
    # 这里需要重构 api_server 来使用 manager
    
    return api


def start_ui_server(ui_app, host, port):
    """启动 UI 服务器"""
    logger.info(f"启动 Gradio UI: http://{host}:{port}")
    ui_app.launch(
        server_name=host,
        server_port=port,
        share=False,
        show_error=True
    )


def start_api_server(api_app, host, port):
    """启动 API 服务器"""
    logger.info(f"启动 FastAPI: http://{host}:{port}")
    uvicorn.run(
        api_app,
        host=host,
        port=port,
        log_level="info"
    )


def main():
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("Step-Audio-EditX 统一服务器")
    logger.info("=" * 80)
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"懒加载: 启用 (空闲超时: {args.idle_timeout}秒)")
    logger.info(f"自动卸载: {'禁用' if args.disable_auto_unload else '启用'}")
    logger.info("=" * 80)
    
    # 初始化共享管理器
    manager = SharedModelManager(args)
    
    # 创建应用
    ui_app = create_ui(manager, args)
    api_app = create_api(manager)
    
    # 在单独线程启动 API
    api_thread = threading.Thread(
        target=start_api_server,
        args=(api_app, args.host, args.api_port),
        daemon=True
    )
    api_thread.start()
    
    # 主线程启动 UI
    logger.info("=" * 80)
    logger.info(f"UI: http://{args.host}:{args.ui_port}")
    logger.info(f"API: http://{args.host}:{args.api_port}")
    logger.info(f"API 文档: http://{args.host}:{args.api_port}/docs")
    logger.info("=" * 80)
    
    try:
        start_ui_server(ui_app, args.host, args.ui_port)
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
    finally:
        for variant, mgr in manager.model_managers.items():
            mgr.shutdown()


if __name__ == "__main__":
    main()
