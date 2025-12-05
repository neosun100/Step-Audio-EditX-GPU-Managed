"""
Unified Server - 统一服务器
同时提供 Gradio UI 和 FastAPI，共享同一个模型实例
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

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
    parser = argparse.ArgumentParser(description="Step-Audio-EditX 统一服务器 (UI + API)")
    
    # 模型配置
    parser.add_argument("--model-path", type=str, required=True, help="模型根目录路径")
    parser.add_argument("--model-source", type=str, default="local", choices=["auto", "local", "modelscope", "huggingface"])
    parser.add_argument("--tokenizer-model-id", type=str, default=None)
    parser.add_argument("--torch-dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--enable-auto-transcribe", action="store_true")
    
    # 懒加载配置
    parser.add_argument("--idle-timeout", type=int, default=300, help="模型空闲超时时间（秒），默认300秒")
    parser.add_argument("--disable-auto-unload", action="store_true", help="禁用自动卸载模型")
    
    # 服务器配置
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--ui-port", type=int, default=7860, help="Gradio UI 端口")
    parser.add_argument("--api-port", type=int, default=8000, help="FastAPI 端口")
    
    # 模型变体路径
    parser.add_argument("--awq-model-path", type=str, default=None)
    parser.add_argument("--bnb-model-path", type=str, default=None)
    
    return parser.parse_args()


class UnifiedModelManager:
    """统一模型管理器 - 管理所有模型变体的懒加载"""
    
    def __init__(self, args):
        self.args = args
        self.tokenizer_manager = None
        self.model_managers = {}  # {variant: LazyModelManager}
        self.whisper_asr = None
        
        # 初始化
        self._init_tokenizer()
        self._init_model_variants()
        self._init_whisper()
    
    def _init_tokenizer(self):
        """初始化 Tokenizer（共享，不懒加载）"""
        logger.info("初始化 Audio Tokenizer...")
        
        model_path = Path(self.args.model_path)
        tokenizer_path = model_path / "Step-Audio-Tokenizer"
        
        if not tokenizer_path.exists():
            tokenizer_path = Path(self.args.tokenizer_model_id) if self.args.tokenizer_model_id else tokenizer_path
        
        self.tokenizer = StepAudioTokenizer(
            model_path=str(tokenizer_path),
            model_source=self.args.model_source
        )
        logger.info("Audio Tokenizer 初始化完成")
    
    def _init_model_variants(self):
        """初始化所有模型变体的懒加载管理器"""
        model_path = Path(self.args.model_path)
        
        # Base 模型
        base_path = model_path / "Step-Audio-EditX"
        if base_path.exists():
            logger.info(f"注册 base 模型: {base_path}")
            self.model_managers["base"] = LazyModelManager(
                model_factory=lambda: self._create_tts_model(str(base_path), None),
                idle_timeout=self.args.idle_timeout,
                auto_unload=not self.args.disable_auto_unload
            )
        
        # AWQ 模型
        awq_path = Path(self.args.awq_model_path) if self.args.awq_model_path else model_path / "Step-Audio-EditX-AWQ-4bit"
        if awq_path.exists():
            logger.info(f"注册 awq 模型: {awq_path}")
            self.model_managers["awq"] = LazyModelManager(
                model_factory=lambda: self._create_tts_model(str(awq_path), "awq-4bit"),
                idle_timeout=self.args.idle_timeout,
                auto_unload=not self.args.disable_auto_unload
            )
        
        # BnB 模型
        bnb_path = Path(self.args.bnb_model_path) if self.args.bnb_model_path else model_path / "Step-Audio-EditX-bnb-4bit"
        if bnb_path.exists():
            logger.info(f"注册 bnb 模型: {bnb_path}")
            self.model_managers["bnb"] = LazyModelManager(
                model_factory=lambda: self._create_tts_model(str(bnb_path), "int4"),
                idle_timeout=self.args.idle_timeout,
                auto_unload=not self.args.disable_auto_unload
            )
        
        logger.info(f"已注册 {len(self.model_managers)} 个模型变体: {list(self.model_managers.keys())}")
    
    def _create_tts_model(self, model_path: str, quantization: Optional[str]):
        """创建 TTS 模型实例"""
        torch_dtype = getattr(torch, self.args.torch_dtype)
        
        return StepAudioTTS(
            model_path=model_path,
            audio_tokenizer=self.tokenizer,
            model_source=self.args.model_source,
            quantization_config=quantization,
            torch_dtype=torch_dtype,
            device_map=self.args.device_map
        )
    
    def _init_whisper(self):
        """初始化 Whisper ASR（如果启用）"""
        if self.args.enable_auto_transcribe:
            logger.info("初始化 Whisper ASR...")
            self.whisper_asr = WhisperWrapper()
            logger.info("Whisper ASR 初始化完成")
    
    def get_model(self, variant: str = "base") -> StepAudioTTS:
        """获取指定变体的模型（懒加载）"""
        if variant not in self.model_managers:
            available = list(self.model_managers.keys())
            raise ValueError(f"模型变体 '{variant}' 不存在。可用: {available}")
        
        return self.model_managers[variant].get_model()
    
    def get_status(self):
        """获取所有模型的状态"""
        status = {}
        for variant, manager in self.model_managers.items():
            status[variant] = manager.get_status()
        return status
    
    def shutdown(self):
        """关闭所有模型管理器"""
        logger.info("正在关闭所有模型管理器...")
        for variant, manager in self.model_managers.items():
            logger.info(f"关闭 {variant} 模型...")
            manager.shutdown()
        logger.info("所有模型管理器已关闭")


def create_gradio_ui(model_manager: UnifiedModelManager):
    """创建 Gradio UI"""
    # 导入 app.py 中的 UI 组件
    # 这里简化处理，实际需要重构 app.py
    from app import create_interface
    
    # 创建接口，传入模型管理器
    interface = create_interface(model_manager)
    return interface


def create_fastapi_app(model_manager: UnifiedModelManager):
    """创建 FastAPI 应用"""
    from api_server import build_fastapi_app
    from api.voices import list_presets
    
    # 构建 API 应用
    app = FastAPI(
        title="Step-Audio-EditX Unified API",
        version="2.0.0",
        description="统一的 TTS 和音频编辑 API，支持懒加载"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 存储模型管理器
    app.state.model_manager = model_manager
    app.state.generate_lock = asyncio.Lock()
    
    # 健康检查
    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}
    
    # 模型状态
    @app.get("/v1/models/status")
    async def models_status():
        return model_manager.get_status()
    
    # 导入其他 API 端点
    from api_server import build_fastapi_app
    # 这里需要重构 api_server.py 来使用 model_manager
    
    return app


def main():
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("Step-Audio-EditX 统一服务器启动")
    logger.info("=" * 80)
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"UI 端口: {args.ui_port}")
    logger.info(f"API 端口: {args.api_port}")
    logger.info(f"懒加载: 启用 (空闲超时: {args.idle_timeout}秒)")
    logger.info(f"自动卸载: {'禁用' if args.disable_auto_unload else '启用'}")
    logger.info("=" * 80)
    
    # 初始化统一模型管理器
    model_manager = UnifiedModelManager(args)
    
    # 创建 Gradio UI
    logger.info("创建 Gradio UI...")
    # ui_app = create_gradio_ui(model_manager)
    
    # 创建 FastAPI
    logger.info("创建 FastAPI...")
    api_app = create_fastapi_app(model_manager)
    
    # 启动服务器
    # 注意：这里需要使用 gradio 的 mount_gradio_app 来挂载到 FastAPI
    # 或者使用多线程分别启动
    
    logger.info("=" * 80)
    logger.info("服务器启动完成！")
    logger.info(f"Gradio UI: http://{args.host}:{args.ui_port}")
    logger.info(f"FastAPI: http://{args.host}:{args.api_port}")
    logger.info(f"API 文档: http://{args.host}:{args.api_port}/docs")
    logger.info("=" * 80)
    
    try:
        # 启动 FastAPI
        uvicorn.run(
            api_app,
            host=args.host,
            port=args.api_port,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
    finally:
        model_manager.shutdown()


if __name__ == "__main__":
    main()
