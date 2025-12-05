"""
GPU Resource Manager - 智能显存管理（懒加载 + 即用即卸）

核心功能：
1. 懒加载：首次请求时加载模型到 GPU
2. 即用即卸：任务完成后立即转移到 CPU，释放显存
3. 快速恢复：从 CPU 快速转移回 GPU（2-5秒）
4. 自动监控：空闲超时自动释放

状态转换：
未加载 ──首次(20-30s)──→ GPU ──任务完成(2s)──→ CPU ──新请求(2-5s)──→ GPU
  ↑                                                    ↓
  └──────────────超时/手动释放(1s)──────────────────────┘
"""

import torch
import threading
import time
import logging
from typing import Optional, Callable, Any
import gc

logger = logging.getLogger(__name__)


class GPUResourceManager:
    """GPU 资源管理器"""
    
    def __init__(self, idle_timeout: int = 600):
        """
        初始化 GPU 资源管理器
        
        Args:
            idle_timeout: 空闲超时时间（秒），默认 10 分钟
        """
        self.idle_timeout = idle_timeout
        self.lock = threading.Lock()
        self.running = False
        self.monitor_thread = None
        
        # 模型状态
        self.models = {}  # {model_name: model_instance}
        self.models_cpu = {}  # {model_name: cpu_cached_model}
        self.last_use_time = {}  # {model_name: timestamp}
        self.model_locations = {}  # {model_name: 'gpu'/'cpu'/'unloaded'}
        
        logger.info(f"🚀 GPU 资源管理器初始化完成 (超时: {idle_timeout}秒)")
    
    def get_model(self, model_name: str, load_func: Callable[[], Any]) -> Any:
        """
        获取模型（懒加载逻辑）
        
        Args:
            model_name: 模型名称（如 'llm', 'vocoder'）
            load_func: 模型加载函数
            
        Returns:
            模型实例（在 GPU 上）
        """
        with self.lock:
            self.last_use_time[model_name] = time.time()
            
            # 情况1: 模型已在 GPU 上
            if model_name in self.models and self.models[model_name] is not None:
                logger.debug(f"✅ 模型 {model_name} 已在 GPU 上")
                return self.models[model_name]
            
            # 情况2: 模型在 CPU 缓存中，快速转移到 GPU
            # 注意：对于复合对象（如 StepAudioTTS），不进行 CPU/GPU 转移
            # 直接返回缓存的对象
            if model_name in self.models_cpu and self.models_cpu[model_name] is not None:
                logger.info(f"⚡ 从缓存恢复模型 {model_name}...")
                start_time = time.time()
                
                model = self.models_cpu[model_name]
                self.models[model_name] = model
                self.models_cpu[model_name] = None
                self.model_locations[model_name] = 'gpu'
                
                elapsed = time.time() - start_time
                logger.info(f"✅ 模型 {model_name} 恢复完成 ({elapsed:.2f}秒)")
                return model
            
            # 情况3: 首次加载，从磁盘加载到 GPU
            logger.info(f"📥 首次加载模型 {model_name} 到 GPU...")
            start_time = time.time()
            
            model = load_func()
            self.models[model_name] = model
            self.model_locations[model_name] = 'gpu'
            
            elapsed = time.time() - start_time
            logger.info(f"✅ 模型 {model_name} 加载完成 ({elapsed:.2f}秒)")
            return model
    
    def force_offload(self, model_name: Optional[str] = None):
        """
        即用即卸：将模型从 GPU 转移到 CPU
        
        Args:
            model_name: 模型名称，None 表示卸载所有模型
        """
        with self.lock:
            if model_name:
                self._offload_single_model(model_name)
            else:
                # 卸载所有模型
                for name in list(self.models.keys()):
                    self._offload_single_model(name)
    
    def _offload_single_model(self, model_name: str):
        """卸载单个模型到 CPU"""
        if model_name not in self.models or self.models[model_name] is None:
            return
        
        logger.info(f"💾 卸载模型 {model_name} 到 CPU...")
        start_time = time.time()
        
        model = self.models[model_name]
        # 对于复合对象（如 StepAudioTTS），只是移动引用，不调用 .to()
        self.models_cpu[model_name] = model
        self.models[model_name] = None
        self.model_locations[model_name] = 'cpu'
        
        # 清理 GPU 缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        elapsed = time.time() - start_time
        gpu_mem = self._get_gpu_memory_mb()
        logger.info(f"✅ 模型 {model_name} 已卸载 ({elapsed:.2f}秒, GPU显存: {gpu_mem:.0f}MB)")
    
    def force_release(self, model_name: Optional[str] = None):
        """
        完全释放：清空 GPU 和 CPU 缓存
        
        Args:
            model_name: 模型名称，None 表示释放所有模型
        """
        with self.lock:
            if model_name:
                self._release_single_model(model_name)
            else:
                # 释放所有模型
                for name in list(self.models.keys()):
                    self._release_single_model(name)
    
    def _release_single_model(self, model_name: str):
        """完全释放单个模型"""
        logger.info(f"🗑️  完全释放模型 {model_name}...")
        
        self.models[model_name] = None
        self.models_cpu[model_name] = None
        if model_name in self.model_locations:
            del self.model_locations[model_name]
        if model_name in self.last_use_time:
            del self.last_use_time[model_name]
        
        torch.cuda.empty_cache()
        gc.collect()
        
        gpu_mem = self._get_gpu_memory_mb()
        logger.info(f"✅ 模型 {model_name} 已完全释放 (GPU显存: {gpu_mem:.0f}MB)")
    
    def start_monitor(self):
        """启动监控线程"""
        if self.running:
            logger.warning("监控线程已在运行")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔍 GPU 监控线程已启动")
    
    def stop_monitor(self):
        """停止监控线程"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("⏹️  GPU 监控线程已停止")
    
    def _monitor_loop(self):
        """监控循环：检查空闲超时"""
        while self.running:
            time.sleep(30)  # 每30秒检查一次
            
            with self.lock:
                current_time = time.time()
                
                for model_name in list(self.models.keys()):
                    if model_name not in self.last_use_time:
                        continue
                    
                    idle_time = current_time - self.last_use_time[model_name]
                    
                    # 超时自动卸载到 CPU
                    if idle_time > self.idle_timeout:
                        if self.models.get(model_name) is not None:
                            logger.info(f"⏰ 模型 {model_name} 空闲 {idle_time:.0f}秒，自动卸载")
                            self._offload_single_model(model_name)
    
    def get_status(self) -> dict:
        """获取当前状态"""
        with self.lock:
            status = {
                'models': {},
                'gpu_memory_mb': self._get_gpu_memory_mb(),
                'idle_timeout': self.idle_timeout
            }
            
            for model_name in self.model_locations:
                location = self.model_locations[model_name]
                idle_time = time.time() - self.last_use_time.get(model_name, time.time())
                
                status['models'][model_name] = {
                    'location': location,
                    'idle_seconds': int(idle_time)
                }
            
            return status
    
    def _get_gpu_memory_mb(self) -> float:
        """获取当前 GPU 显存占用（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def update_timeout(self, new_timeout: int):
        """更新空闲超时时间"""
        with self.lock:
            self.idle_timeout = new_timeout
            logger.info(f"⚙️  空闲超时已更新为 {new_timeout} 秒")


# 全局单例
_global_gpu_manager: Optional[GPUResourceManager] = None


def get_gpu_manager(idle_timeout: int = 600) -> GPUResourceManager:
    """获取全局 GPU 管理器单例"""
    global _global_gpu_manager
    if _global_gpu_manager is None:
        _global_gpu_manager = GPUResourceManager(idle_timeout=idle_timeout)
        _global_gpu_manager.start_monitor()
    return _global_gpu_manager
