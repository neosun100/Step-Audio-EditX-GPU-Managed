"""
Lazy Model Manager - 懒加载模型管理器
支持自动加载、卸载和显存释放
"""
import logging
import threading
import time
import gc
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)


class LazyModelManager:
    """
    懒加载模型管理器
    
    特性：
    - 需要时才加载模型
    - 空闲一定时间后自动卸载
    - 释放GPU显存
    - 线程安全
    """
    
    def __init__(
        self,
        model_factory,
        idle_timeout: int = 300,  # 5分钟空闲后卸载
        auto_unload: bool = True
    ):
        """
        初始化懒加载管理器
        
        Args:
            model_factory: 模型工厂函数，返回模型实例
            idle_timeout: 空闲超时时间（秒），默认300秒
            auto_unload: 是否自动卸载，默认True
        """
        self.model_factory = model_factory
        self.idle_timeout = idle_timeout
        self.auto_unload = auto_unload
        
        self._model = None
        self._lock = threading.RLock()
        self._last_access_time = 0
        self._monitor_thread = None
        self._stop_monitor = threading.Event()
        
        if self.auto_unload:
            self._start_monitor()
    
    def _start_monitor(self):
        """启动监控线程"""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_monitor.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_idle,
                daemon=True,
                name="LazyModelMonitor"
            )
            self._monitor_thread.start()
            logger.info(f"模型监控线程已启动，空闲超时: {self.idle_timeout}秒")
    
    def _monitor_idle(self):
        """监控模型空闲状态"""
        while not self._stop_monitor.is_set():
            time.sleep(10)  # 每10秒检查一次
            
            with self._lock:
                if self._model is not None and self._last_access_time > 0:
                    idle_time = time.time() - self._last_access_time
                    if idle_time > self.idle_timeout:
                        logger.info(f"模型空闲 {idle_time:.1f}秒，开始卸载...")
                        self._unload_model()
    
    def _load_model(self):
        """加载模型"""
        if self._model is None:
            logger.info("正在加载模型...")
            start_time = time.time()
            self._model = self.model_factory()
            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
        self._last_access_time = time.time()
    
    def _unload_model(self):
        """卸载模型并释放显存"""
        if self._model is not None:
            logger.info("正在卸载模型...")
            
            # 删除模型引用
            del self._model
            self._model = None
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 强制垃圾回收
            gc.collect()
            
            # 显示显存使用情况
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"显存使用: {allocated:.2f}GB (已分配) / {reserved:.2f}GB (已保留)")
            
            logger.info("模型已卸载，显存已释放")
            self._last_access_time = 0
    
    def get_model(self):
        """
        获取模型实例（懒加载）
        
        Returns:
            模型实例
        """
        with self._lock:
            self._load_model()
            return self._model
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        with self._lock:
            return self._model is not None
    
    def force_unload(self):
        """强制卸载模型"""
        with self._lock:
            self._unload_model()
    
    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        with self._lock:
            status = {
                "loaded": self._model is not None,
                "auto_unload": self.auto_unload,
                "idle_timeout": self.idle_timeout,
            }
            
            if self._model is not None and self._last_access_time > 0:
                idle_time = time.time() - self._last_access_time
                status["idle_time"] = idle_time
                status["time_until_unload"] = max(0, self.idle_timeout - idle_time)
            
            if torch.cuda.is_available():
                status["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
                status["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            
            return status
    
    def shutdown(self):
        """关闭管理器"""
        logger.info("正在关闭模型管理器...")
        self._stop_monitor.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5)
        self.force_unload()
        logger.info("模型管理器已关闭")
    
    def __del__(self):
        """析构函数"""
        try:
            self.shutdown()
        except:
            pass
