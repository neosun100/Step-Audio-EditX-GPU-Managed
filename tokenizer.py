import io
import threading
import time
import os
import hashlib
import json
from pathlib import Path
import logging

import numpy as np
import torch
import torchaudio
import onnxruntime
import whisper

from funasr_detach import AutoModel
from utils import resample_audio, energy_norm_fn, trim_silence
from model_loader import model_loader, ModelSource

logger = logging.getLogger(__name__)


class StepAudioTokenizer:
    def __init__(
        self,
        encoder_path,
        model_source=ModelSource.AUTO,
        funasr_model_id="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
        enable_cache=True,
        cache_max_size=1000,
        enable_gpu_management=False
    ):
        """
        Initialize StepAudioTokenizer

        Args:
            encoder_path: Encoder path
            model_source: Model source
            funasr_model_id: FunASR model ID
            enable_cache: Enable persistent cache
            cache_max_size: Maximum cache size
            enable_gpu_management: Enable GPU memory management
        """
        self.enable_gpu_management = enable_gpu_management
        self.encoder_path = encoder_path
        self.model_source = model_source
        self.funasr_model_id = funasr_model_id
        
        # 延迟加载：不在初始化时加载模型
        self.funasr_model = None
        self.ort_session = None
        self.kms = None
        
        # 保存路径信息，用于懒加载
        self.funasr_model_path = os.path.join(encoder_path, funasr_model_id)
        
        self.chunk_size = [0, 4, 5]
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1

        self.vq02_sessions = {}
        self.vq02_lock = threading.Lock()
        self.vq06_lock = threading.Lock()
        
        # 缓存功能
        self.enable_cache = enable_cache
        self.cache_max_size = cache_max_size
        self.cache_dir = Path("/app/cache/funasr")
        self._cache = {}  # {audio_hash: (speech_tokens, vq02_ori, vq06_ori)}
        self._cache_order = []  # LRU tracking
        self.cache_hits = 0
        self.cache_misses = 0
        
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache_from_disk()
            logger.info(f"🚀 FunASR persistent cache enabled at {self.cache_dir}")
            logger.info(f"   - Max size: {cache_max_size}")
            logger.info(f"   - Loaded: {len(self._cache)} cached items from disk")
        
        logger.info(f"✅ Tokenizer 初始化完成 (懒加载: {enable_gpu_management})")
    
    def _load_models(self):
        """懒加载 FunASR 和 ONNX 模型"""
        if self.funasr_model is not None:
            return
        
        logger.info("📥 首次使用，加载 FunASR 和 ONNX 模型到 GPU...")
        
        # Load FunASR model
        try:
            self.funasr_model = model_loader.load_funasr_model(
                self.encoder_path,
                self.funasr_model_path,
                source=self.model_source,
                model_revision="main"
            )
        except Exception as e:
            logger.warning(f"Failed to load FunASR: {e}")
            self.funasr_model = AutoModel(model=self.funasr_model_path, model_revision="main")
        
        # Load resource files
        kms_path = os.path.join(self.funasr_model.repo_path, "linguistic_tokenizer.npy")
        self.cosy_tokenizer_path = os.path.join(self.funasr_model.repo_path, "speech_tokenizer_v1.onnx")
        
        if not os.path.exists(kms_path):
            raise FileNotFoundError(f"KMS file not found: {kms_path}")
        if not os.path.exists(self.cosy_tokenizer_path):
            raise FileNotFoundError(f"Cosy tokenizer file not found: {self.cosy_tokenizer_path}")
        
        self.kms = torch.tensor(np.load(kms_path))
        
        # 创建 ONNX session（根据 GPU 管理模式选择 provider）
        self._create_ort_session()
        
        logger.info("✅ FunASR 和 ONNX 模型加载完成")
    
    def _create_ort_session(self):
        """创建 ONNX Runtime session"""
        # 如果启用 GPU 管理，使用 CPU provider 避免显存残留
        if self.enable_gpu_management:
            providers = ["CPUExecutionProvider"]
            logger.info("🔧 ONNX 使用 CPU provider（避免显存残留）")
        else:
            providers = ["CUDAExecutionProvider"]
        
        session_option = onnxruntime.SessionOptions()
        session_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_option.intra_op_num_threads = 4
        self.ort_session = onnxruntime.InferenceSession(
            self.cosy_tokenizer_path, sess_options=session_option, providers=providers
        )

    def __call__(self, audio, sr):
        # 懒加载模型
        if self.funasr_model is None:
            self._load_models()
        
        _, vq02, vq06 = self.wav2token(audio, sr, False)
        text = self.merge_vq0206_to_token_str(vq02, vq06)
        return text

    def preprocess_wav(self, audio, sample_rate, enable_trim=True, energy_norm=True):
        audio = resample_audio(audio, sample_rate, 16000)
        if energy_norm:
            audio = energy_norm_fn(audio)

        if enable_trim:
            audio = audio.cpu().numpy().squeeze(0)
            audio = trim_silence(audio, 16000)
            audio = torch.from_numpy(audio)
            audio = audio.unsqueeze(0)
        return audio

    def wav2token(self, audio, sample_rate, enable_trim=True, energy_norm=True):
        audio = self.preprocess_wav(
            audio, sample_rate, enable_trim=enable_trim, energy_norm=energy_norm
        )

        # 🔥 启用缓存逻辑
        if self.enable_cache:
            # 音频已经通过 preprocess_wav 重采样为 16000 Hz
            audio_hash = self._compute_audio_hash(audio, 16000)
            
            # 检查缓存
            cached_result = self._cache_get(audio_hash)
            if cached_result is not None:
                speech_tokens, vq02_ori, vq06_ori = cached_result
                print(f"✅ [FunASR Cache HIT] hash={audio_hash[:8]}... (saved ~1.65s)", flush=True)
                self.cache_hits += 1
                return speech_tokens, vq02_ori, vq06_ori
            
            print(f"❌ [FunASR Cache MISS] hash={audio_hash[:8]}... encoding audio...", flush=True)
            self.cache_misses += 1
            import time
            start_time = time.time()

        # 实际编码
        vq02_ori = self.get_vq02_code(audio)
        vq02 = [int(x) + 65536 for x in vq02_ori]
        vq06_ori = self.get_vq06_code(audio)
        vq06 = [int(x) + 65536 + 1024 for x in vq06_ori]

        chunk = 1
        chunk_nums = min(len(vq06) // (3 * chunk), len(vq02) // (2 * chunk))
        speech_tokens = []
        for idx in range(chunk_nums):
            speech_tokens += vq02[idx * chunk * 2 : (idx + 1) * chunk * 2]
            speech_tokens += vq06[idx * chunk * 3 : (idx + 1) * chunk * 3]
        
        # 缓存结果
        if self.enable_cache:
            encoding_time = time.time() - start_time
            print(f"⏱️  [FunASR Encoding] time={encoding_time:.2f}s, caching result...", flush=True)
            self._cache_set(audio_hash, (speech_tokens, vq02_ori, vq06_ori))
        
        return speech_tokens, vq02_ori, vq06_ori

    def get_vq02_code(self, audio, session_id=None, is_final=True):
        # 确保模型已加载
        if self.funasr_model is None:
            self._load_models()
        
        _tmp_wav = io.BytesIO()
        torchaudio.save(_tmp_wav, audio, 16000, format="wav")
        _tmp_wav.seek(0)

        with self.vq02_lock:
            cache = {}
            if session_id in self.vq02_sessions:
                cache = self.vq02_sessions[session_id].get("cache", {})

            res, new_cache = self.funasr_model.infer_encoder(
                input=[_tmp_wav],
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
                device=0,
                is_final=is_final,
                cache=cache,
            )
            c_list = []
            for j, res_ in enumerate(res):
                feat = res_["enc_out"]
                if len(feat) > 0:
                    c_list = self.dump_label([feat], self.kms)[0]

            if is_final:
                if session_id in self.vq02_sessions:
                    self.vq02_sessions.pop(session_id)
            else:
                if isinstance(session_id, str) and len(session_id) > 0:
                    self.vq02_sessions[session_id] = {
                        "cache": new_cache,
                        "update_time": time.time(),
                    }

            return c_list

    def get_vq06_code(self, audio):
        # 确保模型已加载
        if self.ort_session is None:
            self._load_models()

        def split_audio(audio, chunk_duration=480000):
            start = 0
            chunks = []
            while start < len(audio):
                end = min(start + chunk_duration, len(audio))
                chunk = audio[start:end]
                if len(chunk) < 480:
                    pass
                else:
                    chunks.append(chunk)
                start = end
            return chunks

        with self.vq06_lock:
            audio = audio.squeeze(0)
            chunk_audios = split_audio(audio, chunk_duration=30 * 16000)  # Maximum support 30s
            speech_tokens = []
            for chunk in chunk_audios:
                duration = round(chunk.shape[0] / 16000, 2)
                feat = whisper.log_mel_spectrogram(chunk, n_mels=128)
                feat = feat.unsqueeze(0)
                feat_len = np.array([feat.shape[2]], dtype=np.int32)
                chunk_token = (
                    self.ort_session.run(
                        None,
                        {
                            self.ort_session.get_inputs()[0]
                            .name: feat.detach()
                            .cpu()
                            .numpy(),
                            self.ort_session.get_inputs()[1].name: feat_len,
                        },
                    )[0]
                    .flatten()
                    .tolist()
                )
                assert abs(len(chunk_token) - duration * 25) <= 2
                speech_tokens += chunk_token

            return speech_tokens

    def kmean_cluster(self, samples, means):
        dists = torch.cdist(samples, means)
        indices = dists.argmin(dim=1).cpu().numpy()
        return indices.tolist()

    def dump_label(self, samples, mean):
        dims = samples[0].shape[-1]
        x_lens = [x.shape[1] for x in samples]
        total_len = sum(x_lens)
        x_sel = torch.FloatTensor(1, total_len, dims)
        start_len = 0
        for sample in samples:
            sample_len = sample.shape[1]
            end_len = start_len + sample_len
            x_sel[:, start_len:end_len] = sample
            start_len = end_len
        dense_x = x_sel.squeeze(0)
        indices = self.kmean_cluster(dense_x, mean)
        indices_list = []
        start_len = 0
        for x_len in x_lens:
            end_len = start_len + end_len
            indices_list.append(indices[start_len:end_len])
        return indices_list

    def merge_vq0206_to_token_str(self, vq02, vq06):
        _vq06 = [1024 + x for x in vq06]
        result = []
        i = 0
        j = 0
        while i < len(vq02) - 1 and j < len(_vq06) - 2:
            sublist = vq02[i : i + 2] + _vq06[j : j + 3]
            result.extend(sublist)
            i += 2
            j += 3
        return "".join([f"<audio_{x}>" for x in result])
    
    def _compute_audio_hash(self, audio, sr):
        """计算音频的 MD5 哈希"""
        audio_bytes = audio.cpu().numpy().tobytes() if hasattr(audio, 'cpu') else audio.tobytes()
        hash_str = hashlib.md5(audio_bytes + str(sr).encode()).hexdigest()
        return hash_str
    
    def _get_cache_file_path(self, audio_hash):
        """获取缓存文件路径（两级目录结构）"""
        subdir = self.cache_dir / audio_hash[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{audio_hash}.json"
    
    def _load_cache_from_disk(self):
        """从磁盘加载缓存"""
        if not self.cache_dir.exists():
            return
        
        count = 0
        for cache_file in self.cache_dir.glob("*/*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                audio_hash = data['hash']
                tokens = tuple(data['tokens']) if isinstance(data['tokens'], list) else data['tokens']
                self._cache[audio_hash] = tokens
                self._cache_order.append(audio_hash)
                count += 1
                if count >= self.cache_max_size:
                    break
            except Exception as e:
                logger.debug(f"Failed to load cache file {cache_file}: {e}")
        
        logger.info(f"✅ Loaded {count} cached items from disk")
    
    def _cache_get(self, audio_hash):
        """从缓存获取"""
        if audio_hash in self._cache:
            # LRU: 移到末尾
            self._cache_order.remove(audio_hash)
            self._cache_order.append(audio_hash)
            return self._cache[audio_hash]
        return None
    
    def _cache_set(self, audio_hash, result):
        """存入缓存"""
        # LRU eviction
        if len(self._cache) >= self.cache_max_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
            # 删除磁盘文件
            try:
                oldest_file = self._get_cache_file_path(oldest)
                if oldest_file.exists():
                    oldest_file.unlink()
            except Exception as e:
                logger.debug(f"Failed to delete old cache file: {e}")
        
        self._cache[audio_hash] = result
        self._cache_order.append(audio_hash)
        
        # 持久化到磁盘
        try:
            cache_file = self._get_cache_file_path(audio_hash)
            data = {
                "hash": audio_hash,
                "tokens": result,
                "cached_at": time.time()
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        return {
            "enabled": self.enable_cache,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total_requests": total,
            "hit_rate": f"{hit_rate:.1%}",
            "cache_size": len(self._cache),
            "max_size": self.cache_max_size,
            "time_saved_estimate": f"{self.cache_hits * 1.65:.1f}s"
        }
    
    def clear_cache(self):
        """清空缓存（内存 + 磁盘）"""
        self._cache.clear()
        self._cache_order.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 删除磁盘缓存
        try:
            if self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("🗑️ FunASR cache cleared (memory + disk)")
        except Exception as e:
            logger.warning(f"Failed to clear disk cache: {e}")
    
    def offload_to_cpu(self):
        """卸载 FunASR 模型到 CPU"""
        if not self.enable_gpu_management:
            return
        
        try:
            logger.info("💾 卸载 FunASR 模型到 CPU...")
            
            # 卸载 FunASR 模型
            if hasattr(self, 'funasr_model') and self.funasr_model is not None:
                if hasattr(self.funasr_model, 'model') and hasattr(self.funasr_model.model, 'to'):
                    self.funasr_model.model = self.funasr_model.model.to('cpu')
            
            # ONNX 已经在 CPU 上，无需处理
            
            # 强制垃圾回收和清空 CUDA 缓存
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            logger.info("✅ FunASR 模型已卸载到 CPU")
        except Exception as e:
            logger.warning(f"卸载失败: {e}")
    
    def load_to_gpu(self):
        """加载 FunASR 模型到 GPU"""
        if not self.enable_gpu_management:
            return
        
        try:
            logger.info("⚡ 恢复 FunASR 模型到 GPU...")
            
            # 恢复 FunASR 模型
            if hasattr(self, 'funasr_model') and self.funasr_model is not None:
                if hasattr(self.funasr_model, 'model') and hasattr(self.funasr_model.model, 'to'):
                    self.funasr_model.model = self.funasr_model.model.to('cuda')
            
            # ONNX 始终在 CPU 上，无需处理
            
            logger.info("✅ FunASR 模型已恢复到 GPU")
        except Exception as e:
            logger.warning(f"恢复失败: {e}")
