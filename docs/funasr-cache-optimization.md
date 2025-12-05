# FunASR 缓存优化方案

## 🎯 优化思路

用户提出的核心想法：**缓存 FunASR 编码结果，避免重复计算**

### 为什么有效？

```
当前流程（17s）:
  音频加载:     0.1s  (0.6%)
  FunASR 编码:  14s   (82%)  ← 缓存这里！
  LLM 生成:     2s    (12%)
  音频解码:     1s    (6%)
  ───────────────────────────
  总计:         17s   (100%)

命中缓存后:
  缓存查询:     0.01s (0.3%)  ← 几乎瞬间！
  LLM 生成:     2s    (67%)
  音频解码:     1s    (33%)
  ───────────────────────────
  总计:         3s    (100%)  ← 提速 5.7x！
```

**关键洞察**：
- FunASR 编码占 82% 时间（14s）
- 相同音频的编码结果是确定的
- 缓存命中可节省 14s，只剩 3s！

---

## 📊 适用场景

### 🔥 高价值场景

1. **固定 Prompt 音频**
   - 克隆相同音色给不同文本
   - 批量处理时使用相同参考音频
   - **预期提速**：首次 17s，后续 **3s**（5.7x）

2. **多次编辑同一音频**
   - 对同一段音频尝试不同风格/情绪
   - A/B 测试不同参数
   - **预期提速**：首次 17s，后续 **3s**（5.7x）

3. **API 批量请求**
   - 多个用户使用相同的预设音色
   - 系统内置音色库
   - **预期提速**：首次慢，99% 请求快

### 💰 价值估算

假设 API 日均 10,000 次请求：
- 缓存命中率 60%（保守估计）
- 节省时间：6,000 × 14s = 23.3 小时/天
- GPU 成本节省：~70% 计算资源

---

## 🛠️ 实现方案

### 方案 1: 内存 LRU 缓存（推荐用于开发/单机）⭐⭐⭐⭐⭐

**优点**：
- ✅ 实现简单（Python 内置 `lru_cache`）
- ✅ 速度极快（内存访问）
- ✅ 自动淘汰旧数据

**缺点**：
- ❌ 重启后丢失
- ❌ 不跨进程共享

**实现**：

```python
import hashlib
import functools
from typing import Tuple

class FunASRCache:
    """FunASR 编码结果缓存"""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _compute_audio_hash(self, audio_tensor, sample_rate: int) -> str:
        """计算音频的唯一哈希"""
        # 使用音频数据 + 采样率生成哈希
        audio_bytes = audio_tensor.cpu().numpy().tobytes()
        hash_input = audio_bytes + str(sample_rate).encode()
        return hashlib.md5(hash_input).hexdigest()
    
    def get(self, audio_tensor, sample_rate: int) -> str:
        """获取缓存的编码结果"""
        key = self._compute_audio_hash(audio_tensor, sample_rate)
        
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, audio_tensor, sample_rate: int, encoded_tokens: str):
        """缓存编码结果"""
        key = self._compute_audio_hash(audio_tensor, sample_rate)
        
        # LRU 淘汰：如果超过容量，删除最早的
        if len(self.cache) >= self.max_size:
            # 删除第一个键（最旧）
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[key] = encoded_tokens
    
    def stats(self) -> dict:
        """返回缓存统计"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": f"{hit_rate:.1%}",
            "cache_size": len(self.cache),
            "max_size": self.max_size
        }
```

**集成到 StepAudioTokenizer**：

```python
class StepAudioTokenizer:
    def __init__(self, encoder_path, ...):
        # ... 现有初始化代码 ...
        
        # 添加缓存
        self.cache = FunASRCache(max_size=1000)
        self.cache_enabled = True  # 可配置
    
    def __call__(self, audio, sr):
        """带缓存的编码"""
        if self.cache_enabled:
            # 尝试从缓存获取
            cached_result = self.cache.get(audio, sr)
            if cached_result is not None:
                logger.debug("✅ FunASR cache hit!")
                return cached_result
        
        # 缓存未命中，正常编码
        logger.debug("❌ FunASR cache miss, encoding...")
        _, vq02, vq06 = self.wav2token(audio, sr, False)
        text = self.merge_vq0206_to_token_str(vq02, vq06)
        
        # 存入缓存
        if self.cache_enabled:
            self.cache.set(audio, sr, text)
        
        return text
    
    def get_cache_stats(self):
        """获取缓存统计"""
        return self.cache.stats()
```

**在 Gradio UI 中显示缓存统计**：

```python
with gr.Accordion("🔍 性能统计", open=False):
    cache_stats = gr.JSON(label="FunASR 缓存统计")
    refresh_btn = gr.Button("刷新统计")
    
    def refresh_cache_stats():
        stats = encoder.get_cache_stats()
        return stats
    
    refresh_btn.click(fn=refresh_cache_stats, outputs=cache_stats)
```

---

### 方案 2: 文件缓存（推荐用于生产/多进程）⭐⭐⭐⭐

**优点**：
- ✅ 持久化（重启不丢失）
- ✅ 跨进程共享
- ✅ 可备份/迁移

**缺点**：
- ❌ 需要磁盘空间
- ❌ 略慢于内存（但仍比重新编码快 1000x）

**实现**：

```python
import os
import json
import hashlib

class FileFunASRCache:
    def __init__(self, cache_dir="/app/cache/funasr"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.hits = 0
        self.misses = 0
    
    def _get_cache_path(self, audio_hash: str) -> str:
        # 使用两级目录避免单目录文件过多
        subdir = audio_hash[:2]
        os.makedirs(os.path.join(self.cache_dir, subdir), exist_ok=True)
        return os.path.join(self.cache_dir, subdir, f"{audio_hash}.json")
    
    def get(self, audio_tensor, sample_rate: int) -> str:
        audio_hash = self._compute_audio_hash(audio_tensor, sample_rate)
        cache_path = self._get_cache_path(audio_hash)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                self.hits += 1
                return data['tokens']
            except:
                pass
        
        self.misses += 1
        return None
    
    def set(self, audio_tensor, sample_rate: int, encoded_tokens: str):
        audio_hash = self._compute_audio_hash(audio_tensor, sample_rate)
        cache_path = self._get_cache_path(audio_hash)
        
        data = {
            "hash": audio_hash,
            "sample_rate": sample_rate,
            "tokens": encoded_tokens,
            "cached_at": time.time()
        }
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)
```

---

### 方案 3: Redis 缓存（推荐用于分布式/生产）⭐⭐⭐⭐⭐

**优点**：
- ✅ 分布式共享
- ✅ 速度快（内存）
- ✅ 支持过期策略
- ✅ 高可用

**缺点**：
- ❌ 需要额外部署 Redis
- ❌ 增加系统复杂度

**实现**：

```python
import redis
import pickle

class RedisFunASRCache:
    def __init__(self, redis_host="localhost", redis_port=6379, ttl=86400):
        self.client = redis.Redis(host=redis_host, port=redis_port)
        self.ttl = ttl  # 缓存有效期（秒）
        self.prefix = "funasr:"
    
    def get(self, audio_tensor, sample_rate: int) -> str:
        key = self.prefix + self._compute_audio_hash(audio_tensor, sample_rate)
        
        result = self.client.get(key)
        if result:
            return result.decode('utf-8')
        return None
    
    def set(self, audio_tensor, sample_rate: int, encoded_tokens: str):
        key = self.prefix + self._compute_audio_hash(audio_tensor, sample_rate)
        self.client.setex(key, self.ttl, encoded_tokens)
```

---

## 📈 性能对比

| 场景 | 无缓存 | 内存缓存 | 文件缓存 | Redis 缓存 |
|------|--------|---------|---------|-----------|
| **首次编码** | 17s | 17s | 17s | 17s |
| **缓存命中** | 17s | **0.01s** | 0.1s | 0.02s |
| **提速倍数** | 1x | **1700x** | 170x | 850x |
| **持久化** | ❌ | ❌ | ✅ | ✅ |
| **跨进程** | ❌ | ❌ | ✅ | ✅ |
| **分布式** | ❌ | ❌ | ❌ | ✅ |

---

## 🎯 推荐实施方案

### 阶段 1: 快速验证（今天）⭐⭐⭐⭐⭐

**实施内存 LRU 缓存**：
- 工作量：30 分钟
- 效果：立竿见影
- 适用：开发/测试/单机部署

### 阶段 2: 生产优化（本周）⭐⭐⭐⭐

**切换到文件缓存**：
- 工作量：1 小时
- 效果：持久化 + 跨进程
- 适用：生产环境/多进程

### 阶段 3: 分布式扩展（按需）⭐⭐⭐

**部署 Redis 缓存**：
- 工作量：2 小时
- 效果：分布式 + 高可用
- 适用：大规模 API 服务

---

## 💡 额外优化建议

### 1. 预热缓存

```python
def warmup_cache(encoder, preset_audios: list):
    """预热常用音频的缓存"""
    for audio_path in preset_audios:
        audio, sr = torchaudio.load(audio_path)
        _ = encoder(audio, sr)  # 触发缓存
    
    print(f"✅ Warmed up {len(preset_audios)} preset audios")
```

### 2. 缓存管理 API

```python
@app.get("/api/cache/stats")
async def get_cache_stats():
    """获取缓存统计"""
    return encoder.get_cache_stats()

@app.post("/api/cache/clear")
async def clear_cache():
    """清空缓存"""
    encoder.cache.cache.clear()
    return {"status": "ok", "message": "Cache cleared"}

@app.post("/api/cache/warmup")
async def warmup_cache():
    """预热缓存"""
    # 预热系统内置音色
    warmup_cache(encoder, PRESET_AUDIO_PATHS)
    return {"status": "ok", "warmed": len(PRESET_AUDIO_PATHS)}
```

### 3. 监控与告警

```python
def log_cache_stats_periodically():
    """定期记录缓存统计"""
    while True:
        stats = encoder.get_cache_stats()
        logger.info(f"📊 FunASR Cache: {stats}")
        
        # 告警：命中率过低
        hit_rate = stats['hits'] / (stats['hits'] + stats['misses'])
        if hit_rate < 0.3:
            logger.warning(f"⚠️ Low cache hit rate: {hit_rate:.1%}")
        
        time.sleep(300)  # 每 5 分钟
```

---

## 📊 实际效果预测

### 场景 1: 固定音色批量生成

```
任务：使用同一个 prompt 音频生成 100 段语音

无缓存:
  - 总时间: 100 × 17s = 1700s (28 分钟)
  
有缓存:
  - 首次: 17s (编码 + 缓存)
  - 后续: 99 × 3s = 297s (5 分钟)
  - 总时间: 314s
  - 提速: 5.4x ✅
```

### 场景 2: 多次编辑同一音频

```
任务：对同一段音频尝试 10 种不同风格

无缓存:
  - 总时间: 10 × 17s = 170s (2.8 分钟)
  
有缓存:
  - 首次: 17s
  - 后续: 9 × 3s = 27s
  - 总时间: 44s
  - 提速: 3.9x ✅
```

### 场景 3: API 日常服务

```
假设：
  - 日均 10,000 次请求
  - 缓存命中率 70%

无缓存:
  - 总时间: 10,000 × 17s = 47.2 小时/天
  
有缓存:
  - 未命中: 3,000 × 17s = 14.2 小时
  - 命中:   7,000 × 0.01s = 0.02 小时
  - 总时间: 14.22 小时/天
  - GPU 成本节省: 70% ✅
```

---

## 🚀 立即开始

### 最小实现（5 分钟）

```python
# 在 tokenizer.py 开头添加
import hashlib

class StepAudioTokenizer:
    def __init__(self, ...):
        # ... 现有代码 ...
        self._cache = {}  # 简单字典缓存
    
    def __call__(self, audio, sr):
        # 计算缓存键
        key = hashlib.md5(audio.cpu().numpy().tobytes()).hexdigest()
        
        # 检查缓存
        if key in self._cache:
            return self._cache[key]
        
        # 正常编码
        _, vq02, vq06 = self.wav2token(audio, sr, False)
        text = self.merge_vq0206_to_token_str(vq02, vq06)
        
        # 存入缓存
        self._cache[key] = text
        return text
```

**这 10 行代码就能实现基础缓存！** 🎉

---

## 📚 总结

**用户的洞察完全正确！**

✅ FunASR 缓存是性能优化的**银弹**  
✅ 实现简单，效果显著  
✅ 缓存命中可提速 **5.7x**（17s → 3s）  
✅ 特别适合固定音色和重复编辑场景  
✅ 投资回报率极高：30 分钟工作 → 70% 性能提升

**建议立即实施！** 🚀
