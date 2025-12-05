"""
GPU 管理的 TTS 包装器

在不修改原有 TTS 类的情况下，添加 GPU 显存智能管理功能
"""

import logging
from typing import Optional
import torch

from tts import StepAudioTTS
from gpu_manager import get_gpu_manager
from model_loader import ModelSource

logger = logging.getLogger(__name__)


class GPUManagedTTS:
    """
    GPU 管理的 TTS 包装器
    
    功能：
    1. 懒加载：首次调用时才加载模型
    2. 即用即卸：每次调用后自动卸载到 CPU
    3. 快速恢复：从 CPU 快速恢复到 GPU
    """
    
    def __init__(
        self,
        model_path: str,
        audio_tokenizer,
        model_source: str = ModelSource.AUTO,
        tts_model_id: Optional[str] = None,
        quantization_config: Optional[str] = None,
        torch_dtype=torch.bfloat16,
        device_map: str = "cuda",
        gpu_idle_timeout: int = 600,
        enable_gpu_management: bool = True
    ):
        """
        初始化 GPU 管理的 TTS
        
        Args:
            enable_gpu_management: 是否启用 GPU 管理（默认 True）
            gpu_idle_timeout: GPU 空闲超时（秒）
            其他参数同 StepAudioTTS
        """
        self.model_path = model_path
        self.audio_tokenizer = audio_tokenizer
        self.model_source = model_source
        self.tts_model_id = tts_model_id
        self.quantization_config = quantization_config
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        
        self.enable_gpu_management = enable_gpu_management
        self.tts_instance: Optional[StepAudioTTS] = None
        
        if enable_gpu_management:
            self.gpu_manager = get_gpu_manager(idle_timeout=gpu_idle_timeout)
            logger.info(f"✅ GPU 管理已启用 (超时: {gpu_idle_timeout}秒)")
        else:
            self.gpu_manager = None
            # 立即加载模型（传统方式）
            self.tts_instance = self._create_tts_instance()
            logger.info("ℹ️  GPU 管理已禁用，使用传统加载方式")
    
    def _create_tts_instance(self) -> StepAudioTTS:
        """创建 TTS 实例"""
        return StepAudioTTS(
            model_path=self.model_path,
            audio_tokenizer=self.audio_tokenizer,
            model_source=self.model_source,
            tts_model_id=self.tts_model_id,
            quantization_config=self.quantization_config,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map
        )
    
    def _get_tts(self) -> StepAudioTTS:
        """获取 TTS 实例（懒加载）"""
        if not self.enable_gpu_management:
            return self.tts_instance
        
        # 首次加载
        if self.tts_instance is None:
            logger.info("📥 首次加载 TTS 模型...")
            self.tts_instance = self._create_tts_instance()
            logger.info("✅ TTS 模型加载完成")
        else:
            # 从 CPU 恢复到 GPU
            logger.info("⚡ 从 CPU 恢复 TTS 模型到 GPU...")
            try:
                self.tts_instance.load_to_gpu()
                logger.info("✅ TTS 模型已恢复到 GPU")
            except Exception as e:
                logger.error(f"❌ 恢复失败: {e}")
        
        return self.tts_instance
    
    def _offload_after_use(self):
        """使用后卸载（如果启用了 GPU 管理）"""
        if self.enable_gpu_management and self.tts_instance:
            logger.info("💾 卸载 TTS 模型到 CPU...")
            try:
                self.tts_instance.offload_to_cpu()
                logger.info("✅ TTS 模型已卸载到 CPU")
            except Exception as e:
                logger.error(f"❌ 卸载失败: {e}")
    
    def clone(self, prompt_wav_path: str, prompt_text: str, target_text: str, **kwargs):
        """
        语音克隆（带 GPU 管理）
        
        Args:
            同 StepAudioTTS.clone()
        """
        try:
            tts = self._get_tts()
            # 移除不支持的参数
            kwargs.pop('intensity', None)
            result = tts.clone(
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                target_text=target_text,
                **kwargs
            )
            return result
        except Exception as e:
            raise e
        finally:
            # 无论成功还是失败，都要卸载
            self._offload_after_use()
    
    def edit(self, input_audio_path: str, audio_text: str, edit_type: str,
             edit_info: Optional[str] = None, text: Optional[str] = None, **kwargs):
        """
        音频编辑（带 GPU 管理）
        
        Args:
            同 StepAudioTTS.edit()
        """
        try:
            tts = self._get_tts()
            result = tts.edit(
                input_audio_path=input_audio_path,
                audio_text=audio_text,
                edit_type=edit_type,
                edit_info=edit_info,
                text=text
            )
            return result
        except Exception as e:
            raise e
        finally:
            # 无论成功还是失败，都要卸载
            self._offload_after_use()
    
    def get_gpu_status(self) -> dict:
        """获取 GPU 状态"""
        if self.enable_gpu_management and self.gpu_manager:
            return self.gpu_manager.get_status()
        return {'enabled': False}
    
    def force_offload(self):
        """手动卸载到 CPU"""
        if self.enable_gpu_management and self.gpu_manager:
            self.gpu_manager.force_offload('tts')
    
    def force_release(self):
        """完全释放模型"""
        if self.enable_gpu_management and self.gpu_manager:
            self.gpu_manager.force_release('tts')
        else:
            self.tts_instance = None
