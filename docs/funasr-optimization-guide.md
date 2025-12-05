# FunASR 音频编码器优化指南

## 🎯 核心问题

根据实际测试，**FunASR 音频编码器占用 83%（~20s）的处理时间**，这是真正的性能瓶颈！

```
完整 Clone 流程（24s 总计）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ 音频编码 (FunASR):      ~20s  (83%)  ← 优化重点！
  🟢 LLM 生成:               ~2s  (8%)
  🟢 音频解码 (CosyVoice):   ~2s  (8%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**优化 FunASR 比优化 LLM 更有价值！**

如果能将 FunASR 从 20s 优化到 10s：
- 总时间：24s → 14s（提速 **42%**）
- 效果远超任何 LLM 量化方案

---

## 🔍 当前实现分析

### 1. FunASR 编码流程

```python
# tokenizer.py: StepAudioTokenizer.wav2token()
def wav2token(self, audio, sample_rate):
    # 步骤 1: 预处理音频（resample, trim, normalize）
    audio = self.preprocess_wav(audio, sample_rate)
    
    # 步骤 2: VQ02 编码（FunASR Paraformer Encoder）← 瓶颈 1
    vq02_ori = self.get_vq02_code(audio)       # ~10-12s
    
    # 步骤 3: VQ06 编码（Whisper Mel + ONNX）← 瓶颈 2
    vq06_ori = self.get_vq06_code(audio)       # ~8-10s
    
    # 步骤 4: 合并 token
    return merge_tokens(vq02, vq06)
```

### 2. 性能瓶颈定位

| 组件 | 耗时估计 | 占比 | 实现方式 |
|------|---------|------|----------|
| **VQ02 (FunASR)** | ~10-12s | 50% | PyTorch 模型推理 |
| **VQ06 (Whisper)** | ~8-10s | 40% | ONNX Runtime |
| 预处理 | ~1s | 5% | CPU 操作 |
| Token 合并 | ~1s | 5% | CPU 操作 |

### 3. 当前配置

```python
# VQ02: FunASR Paraformer
self.funasr_model = AutoModel(
    model="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
)
# 使用 chunk streaming 模式
chunk_size = [0, 4, 5]

# VQ06: Whisper + ONNX
self.ort_session = onnxruntime.InferenceSession(
    "speech_tokenizer_v1.onnx",
    providers=["CUDAExecutionProvider"]
)
# 优化已启用：
# - graph_optimization_level = ORT_ENABLE_ALL
# - intra_op_num_threads = 1
```

---

## 🚀 优化方案

### 方案 1: ONNX 化 FunASR 模型 ⭐⭐⭐⭐⭐

**原理**：将 PyTorch FunASR 模型转换为 ONNX，使用 TensorRT 加速

**预期提速**：2-3x（10s → 3-5s）

**实施步骤**：

```bash
# 1. 导出 FunASR 模型为 ONNX
python export_funasr_to_onnx.py \
    --model-path /model/Step-Audio-Tokenizer/dengcunqin/... \
    --output funasr_encoder.onnx

# 2. 使用 TensorRT 优化 ONNX 模型
trtexec --onnx=funasr_encoder.onnx \
    --saveEngine=funasr_encoder.trt \
    --fp16 \
    --workspace=4096

# 3. 修改 tokenizer.py 使用 TensorRT 引擎
```

**优点**：
- ✅ 显著提速（2-3x）
- ✅ 不改变模型精度
- ✅ GPU 利用率更高

**缺点**：
- ❌ 需要额外的模型转换工作
- ❌ TensorRT 引擎不跨 GPU 架构通用

**难度**：中等

---

### 方案 2: 批处理优化 ⭐⭐⭐⭐

**原理**：支持 batch inference，一次性处理多个音频

**预期提速**：1.5-2x（单个请求不变，吞吐量提升）

**实施步骤**：

```python
# 修改 tokenizer.py: StepAudioTokenizer
class StepAudioTokenizer:
    def wav2token_batch(self, audios: list[torch.Tensor], sample_rates: list[int]):
        """批处理版本"""
        # 1. 批量预处理
        preprocessed = [
            self.preprocess_wav(audio, sr) 
            for audio, sr in zip(audios, sample_rates)
        ]
        
        # 2. Pad 到相同长度
        max_len = max(a.shape[-1] for a in preprocessed)
        padded = torch.stack([
            torch.nn.functional.pad(a, (0, max_len - a.shape[-1]))
            for a in preprocessed
        ])
        
        # 3. 批量推理 VQ02
        vq02_batch = self.funasr_model.infer_encoder(
            input=padded,  # (batch_size, audio_len)
            batch_size=len(audios)
        )
        
        # 4. 批量推理 VQ06
        mel_features = torch.stack([
            whisper.log_mel_spectrogram(audio)
            for audio in preprocessed
        ])
        vq06_batch = self.ort_session.run(None, {
            "mel_features": mel_features.numpy()
        })
        
        return zip(vq02_batch, vq06_batch)
```

**优点**：
- ✅ 提高吞吐量
- ✅ 更好的 GPU 利用率
- ✅ 适合 API 批量请求

**缺点**：
- ❌ 单个请求延迟不变
- ❌ 需要请求队列和调度器

**难度**：中等

---

### 方案 3: 使用更轻量的编码器 ⭐⭐⭐⭐⭐

**原理**：替换 `paraformer-large` 为更小/更快的模型

**候选模型**：

| 模型 | 参数量 | 预期速度 | 精度损失 |
|------|--------|---------|---------|
| **paraformer-base** | 220M | 2-3x 更快 | 轻微 |
| **whisper-small** | 244M | 3-4x 更快 | 中等 |
| **whisper-tiny** | 39M | 5-6x 更快 | 较大 |

**实施步骤**：

```python
# 1. 修改 tokenizer.py 初始化参数
class StepAudioTokenizer:
    def __init__(
        self,
        encoder_path,
        funasr_model_id="dengcunqin/speech_paraformer-base-asr_nat-zh-cn-16k-common-vocab8404"  # 改用 base
    ):
        self.funasr_model = AutoModel(model=funasr_model_id)
        ...

# 2. 测试精度
python benchmark_encoder_accuracy.py \
    --original-model paraformer-large \
    --candidate-model paraformer-base \
    --test-audio examples/*.wav
```

**优点**：
- ✅ 最简单（只需修改配置）
- ✅ 显著提速（2-4x）
- ✅ 内存占用更小

**缺点**：
- ❌ 可能影响音频质量
- ❌ 需要 A/B 测试验证

**难度**：简单 ⭐

---

### 方案 4: CUDA Graph 加速 ⭐⭐⭐

**原理**：使用 CUDA Graph 捕获固定计算图，减少 kernel launch 开销

**预期提速**：1.3-1.5x

**实施步骤**：

```python
# 修改 tokenizer.py
class StepAudioTokenizer:
    def __init__(self, ...):
        ...
        # 预热 + 捕获 CUDA Graph
        self.vq02_graph = None
        self.vq06_graph = None
        self._warmup_cuda_graphs()
    
    def _warmup_cuda_graphs(self):
        """预热并捕获 CUDA Graph"""
        dummy_audio = torch.randn(1, 16000 * 10).cuda()  # 10s audio
        
        # 预热
        for _ in range(3):
            _ = self.get_vq02_code(dummy_audio)
        
        # 捕获 VQ02 graph
        self.vq02_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.vq02_graph):
            self.vq02_output = self.get_vq02_code(dummy_audio)
        
        # 同样处理 VQ06
        ...
    
    def get_vq02_code(self, audio):
        if self.vq02_graph is not None:
            # 使用 CUDA Graph
            self.vq02_graph.replay()
            return self.vq02_output.clone()
        else:
            # 原始推理
            return self._original_vq02_inference(audio)
```

**优点**：
- ✅ 无精度损失
- ✅ 中等提速（1.3-1.5x）

**缺点**：
- ❌ 只支持固定输入尺寸
- ❌ 实现复杂

**难度**：高

---

### 方案 5: 混合精度 + Torch Compile ⭐⭐⭐⭐

**原理**：使用 PyTorch 2.x 的 `torch.compile()` 优化

**预期提速**：1.5-2x

**实施步骤**：

```python
# 修改 tokenizer.py
class StepAudioTokenizer:
    def __init__(self, ...):
        ...
        # 编译 FunASR 模型
        self.funasr_model = torch.compile(
            self.funasr_model,
            mode="reduce-overhead",  # 或 "max-autotune"
            fullgraph=True
        )
        
        # 启用 TF32 加速（Ampere+ GPU）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
```

**优点**：
- ✅ 简单（只需添加一行代码）
- ✅ 无精度损失
- ✅ PyTorch 原生支持

**缺点**：
- ❌ 首次运行需要编译时间
- ❌ 需要 PyTorch 2.0+

**难度**：简单 ⭐

---

### 方案 6: 多 GPU 并行 ⭐⭐

**原理**：将 VQ02 和 VQ06 分配到不同 GPU 并行计算

**预期提速**：1.8-2x（理论值，实际受限于数据传输）

**实施步骤**：

```python
class StepAudioTokenizer:
    def __init__(self, encoder_path, vq02_device="cuda:0", vq06_device="cuda:1"):
        # VQ02 在 GPU 0
        self.funasr_model = AutoModel(...).to(vq02_device)
        
        # VQ06 在 GPU 1
        self.ort_session = onnxruntime.InferenceSession(
            ...,
            providers=[("CUDAExecutionProvider", {"device_id": 1})]
        )
    
    def wav2token(self, audio, sr):
        audio = self.preprocess_wav(audio, sr)
        
        # 并行执行
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_vq02 = executor.submit(self.get_vq02_code, audio)
            future_vq06 = executor.submit(self.get_vq06_code, audio)
            
            vq02 = future_vq02.result()
            vq06 = future_vq06.result()
        
        return self.merge_vq0206_to_token_str(vq02, vq06)
```

**优点**：
- ✅ 充分利用多 GPU
- ✅ 并行计算，理论 2x 提速

**缺点**：
- ❌ 需要额外的 GPU
- ❌ 数据传输开销
- ❌ 不适合单 GPU 用户

**难度**：中等

---

## 📊 优化方案对比

| 方案 | 预期提速 | 实施难度 | 精度影响 | GPU 需求 | 推荐度 |
|------|---------|---------|---------|---------|--------|
| **ONNX + TensorRT** | 2-3x | 中 | 无 | 1 GPU | ⭐⭐⭐⭐⭐ |
| **轻量模型** | 2-4x | 低 | 轻微 | 1 GPU | ⭐⭐⭐⭐⭐ |
| **Torch Compile** | 1.5-2x | 低 | 无 | 1 GPU | ⭐⭐⭐⭐ |
| **批处理** | 1.5-2x | 中 | 无 | 1 GPU | ⭐⭐⭐⭐ |
| **CUDA Graph** | 1.3-1.5x | 高 | 无 | 1 GPU | ⭐⭐⭐ |
| **多 GPU 并行** | 1.8-2x | 中 | 无 | 2+ GPU | ⭐⭐ |

---

## 🎯 推荐实施路线图

### 阶段 1: 快速优化（1-2 天）⭐⭐⭐⭐⭐

**方案**：轻量模型 + Torch Compile

```bash
# 1. 修改配置使用 paraformer-base
vim tokenizer.py  # 修改 funasr_model_id

# 2. 添加 torch.compile()
# 在 __init__ 中添加：
self.funasr_model = torch.compile(self.funasr_model)

# 3. 测试
python benchmark_encoder.py --model base --compile
```

**预期效果**：
- 20s → 8-10s（提速 **2-2.5x**）
- 总时间：24s → 12-14s（提速 **42-50%**）

---

### 阶段 2: 深度优化（1-2 周）⭐⭐⭐⭐⭐

**方案**：ONNX + TensorRT

```bash
# 1. 导出 ONNX
python scripts/export_funasr_onnx.py

# 2. TensorRT 优化
trtexec --onnx=funasr.onnx --saveEngine=funasr.trt --fp16

# 3. 集成 TensorRT 引擎
python scripts/integrate_tensorrt.py

# 4. 测试
python benchmark_encoder.py --engine tensorrt
```

**预期效果**：
- 20s → 6-8s（提速 **2.5-3.3x**）
- 总时间：24s → 10-12s（提速 **50-58%**）

---

### 阶段 3: 生产优化（持续）⭐⭐⭐⭐

**方案**：批处理 + 请求队列

```bash
# 1. 实现批处理 API
vim api_server.py  # 添加请求队列

# 2. 动态批处理
python api_server.py \
    --enable-batching \
    --max-batch-size 4 \
    --batch-timeout 100ms

# 3. 压测
locust -f load_test.py --host http://localhost:8003
```

**预期效果**：
- 吞吐量：1 req/s → 3-4 req/s（提速 **3-4x**）
- 适合高并发场景

---

## 🛠️ 立即可做的优化

### 1. 启用 TF32（30 秒完成）⭐⭐⭐⭐⭐

```python
# 在 tokenizer.py __init__ 开头添加：
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**预期提速**：1.1-1.2x（免费的性能）

### 2. 减少不必要的数据传输

```python
# 当前实现（慢）:
audio = audio.cpu().numpy()  # GPU → CPU
audio = trim_silence(audio)
audio = torch.from_numpy(audio).cuda()  # CPU → GPU

# 优化后（快）:
if audio.is_cuda:
    audio = trim_silence_gpu(audio)  # 全程 GPU
```

### 3. 使用更优的 ONNX Runtime 配置

```python
# 当前：
session_option.intra_op_num_threads = 1  # 太保守！

# 优化：
session_option.intra_op_num_threads = 4
session_option.inter_op_num_threads = 2
session_option.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
```

---

## 📈 预期总体优化效果

```
当前基线（Base 模型）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  音频编码:  20s  (83%)
  LLM 生成:   2s  (8%)
  音频解码:   2s  (8%)
  ─────────────────────────────────────
  总计:      24s  (100%)

阶段 1 优化（轻量模型 + Torch Compile）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  音频编码:   8s  (67%)  ← 提速 2.5x
  LLM 生成:   2s  (17%)
  音频解码:   2s  (17%)
  ─────────────────────────────────────
  总计:      12s  ← 提速 50% ✅

阶段 2 优化（ONNX + TensorRT）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  音频编码:   6s  (60%)  ← 提速 3.3x
  LLM 生成:   2s  (20%)
  音频解码:   2s  (20%)
  ─────────────────────────────────────
  总计:      10s  ← 提速 58% ✅✅

理想情况（所有优化）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  音频编码:   4s  (50%)  ← 提速 5x
  LLM 生成:   2s  (25%)
  音频解码:   2s  (25%)
  ─────────────────────────────────────
  总计:       8s  ← 提速 66% ✅✅✅
```

---

## 🚀 下一步行动

### 立即执行（今天）：

1. **启用 TF32**（30 秒）
   ```python
   # tokenizer.py:18 (在 import 后)
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   ```

2. **测试轻量模型**（30 分钟）
   ```bash
   # 下载 paraformer-base
   git lfs clone https://huggingface.co/damo/speech_paraformer-base_asr_nat-zh-cn-16k-common-vocab8404
   
   # 修改配置
   vim tokenizer.py  # 改 model_id
   
   # 测试
   python test_clone.py --measure-time
   ```

### 本周完成：

3. **添加 Torch Compile**（1 天）
4. **优化 ONNX Runtime 配置**（1 天）
5. **性能基准测试**（1 天）

### 下周开始：

6. **ONNX 导出 + TensorRT 优化**（1-2 周）
7. **批处理实现**（1 周）

---

## 📝 总结

**关键洞察**：
- ✅ FunASR 编码占 83% 时间，是真正的瓶颈
- ✅ 优化编码器比优化 LLM 更有价值（10x）
- ✅ 最简单的方案（轻量模型）就能提速 2-3x
- ✅ 结合多种方案，理论可提速 5x+

**推荐路径**：
1. 先用轻量模型快速验证效果
2. 确认精度可接受后，部署到生产
3. 长期投入 TensorRT 优化获得最佳性能

**投资回报**：
- 阶段 1：2 天工作 → 50% 性能提升
- 阶段 2：2 周工作 → 60% 性能提升
- ROI 极高！🚀
