import logging
import torch
import torchaudio
from transformers import pipeline


class WhisperWrapper:
    """Simplified Whisper ASR wrapper"""

    def __init__(self, model_id="openai/whisper-large-v3-turbo", enable_gpu_management=False):
        """
        Initialize WhisperWrapper

        Args:
            model_id: Whisper model ID, default uses openai/whisper-large-v3-turbo (faster, 50% smaller than v3)
            enable_gpu_management: Enable GPU memory management (lazy loading)
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_id = model_id
        self.enable_gpu_management = enable_gpu_management

        if not enable_gpu_management:
            # 传统方式：立即加载
            self._load_model()
        else:
            self.logger.info(f"✓ Whisper 懒加载已启用")

    def _load_model(self):
        """加载模型到 GPU"""
        if self.model is not None:
            return
        
        try:
            self.logger.info(f"📥 加载 Whisper 模型到 GPU...")
            self.model = pipeline("automatic-speech-recognition", model=self.model_id, device="cuda")
            self.logger.info(f"✓ Whisper model loaded successfully: {self.model_id} (using GPU)")
        except Exception as e:
            self.logger.error(f"❌ Failed to load Whisper model: {e}")
            raise
    
    def offload_to_cpu(self):
        """卸载模型到 CPU"""
        if self.model is not None and hasattr(self.model, 'model'):
            self.logger.info("💾 卸载 Whisper 模型到 CPU...")
            self.model.model = self.model.model.to('cpu')
            torch.cuda.empty_cache()
            self.logger.info("✅ Whisper 模型已卸载到 CPU")
    
    def load_to_gpu(self):
        """加载模型到 GPU"""
        if self.model is not None and hasattr(self.model, 'model'):
            self.logger.info("⚡ 恢复 Whisper 模型到 GPU...")
            self.model.model = self.model.model.to('cuda')
            self.logger.info("✅ Whisper 模型已恢复到 GPU")

    def __call__(self, audio_input):
        """
        Audio to text transcription

        Args:
            audio_input: Audio file path or audio tensor

        Returns:
            Transcribed text
        """
        try:
            # 懒加载
            if self.enable_gpu_management and self.model is None:
                self._load_model()
            elif self.enable_gpu_management and self.model is not None:
                self.load_to_gpu()
            
            if self.model is None:
                raise RuntimeError("Whisper model not loaded")

            # 处理音频
            result = self._transcribe(audio_input)
            
            return result
        finally:
            # 卸载
            if self.enable_gpu_management:
                self.offload_to_cpu()
    
    def _transcribe(self, audio_input):
        """
        执行转写
        
        Args:
            audio_input: Audio file path or audio tensor

        Returns:
            Transcribed text
        """
        try:
            # Load audio
            if isinstance(audio_input, str):
                # Audio file path
                audio, audio_sr = torchaudio.load(audio_input)
                audio = torchaudio.functional.resample(audio, audio_sr, 16000)
                # Handle stereo to mono conversion (pipeline may not handle this)
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)  # Convert stereo to mono by averaging
                # Convert to numpy and squeeze
                audio = audio.squeeze(0).numpy()
            elif isinstance(audio_input, torch.Tensor):
                # Tensor input
                audio = audio_input.cpu()
                audio = torchaudio.functional.resample(audio, audio_sr, 16000)
                # Handle stereo to mono conversion
                if audio.ndim > 1 and audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                audio = audio.squeeze().numpy()
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

            # Transcribe
            result = self.model(audio)
            text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()

            self.logger.debug(f"Transcription result: {text}")
            return text

        except Exception as e:
            self.logger.error(f"Audio transcription failed: {e}")
            return ""

    def is_available(self):
        """Check if whisper model is available"""
        return self.model is not None