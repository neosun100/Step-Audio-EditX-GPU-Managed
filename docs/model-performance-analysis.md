# Step-Audio-EditX 模型性能分析报告

## 📊 测试结果概览

| 指标 | Base 模型 | AWQ 4-bit 模型 | 对比 |
|------|-----------|----------------|------|
| **模型大小** | 16 GB (权重 6.6GB) | 7.1 GB (权重 2.4GB) | AWQ 节省 **56% 磁盘空间** |
| **加载时间** | 2.31s | 2.77s | AWQ 慢 20% |
| **推理时间** | 1.486s | 5.861s | **AWQ 慢 3.94 倍** ⚠️ |
| **显存占用** | ~23-24 GB | ~23-24 GB | 相近 |

## 🔍 问题根因分析

### 1. 量化格式问题

AWQ 模型使用的是 **compressed-tensors** 格式，而非传统的 AutoAWQ 格式：

```json
{
  "quant_method": "compressed-tensors",
  "quantization_status": "compressed",
  "config_groups": {
    "group_0": {
      "format": "pack-quantized",
      "weights": {
        "num_bits": 4,
        "group_size": 128,
        "symmetric": true,
        "type": "int",
        "dynamic": false
      }
    }
  }
}
```

**问题**：
- `compressed-tensors` 是相对较新的量化格式，优化不如成熟的 AutoAWQ
- 缺少专门的 CUDA kernel 加速
- 动态反量化（dequantization）开销大

### 2. 推理路径对比

#### Base 模型推理路径
```
输入 → BFloat16 矩阵运算 → 输出
       └─ 使用优化的 cuBLAS/Flash Attention
```

#### AWQ 模型推理路径（compressed-tensors）
```
输入 → 4-bit 权重读取 → 动态反量化到 BFloat16 → 矩阵运算 → 输出
       └─ 每次前向传播都要反量化    └─ 通用 Python 代码，无专用 kernel
       └─ CPU-GPU 数据传输开销
```

### 3. 为什么 AutoAWQ 格式会更快？

AutoAWQ 使用：
- **融合 kernel**：量化权重直接参与矩阵乘法，无需完整反量化
- **Group-wise 量化优化**：硬件级别的并行处理
- **预编译 CUDA kernel**：针对 NVIDIA GPU 深度优化
- **零拷贝推理**：减少 CPU-GPU 数据传输

compressed-tensors 缺少这些优化，导致：
- 每层都要动态反量化（bottleneck）
- 使用通用 PyTorch 操作，无专用加速
- 额外的内存拷贝和类型转换开销

## 💡 解决方案

### 方案 1：使用 AutoAWQ 重新量化（推荐）

```bash
# 安装 AutoAWQ
pip install autoawq

# 重新量化模型
python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = '/model/Step-Audio-EditX'
quant_path = '/model/Step-Audio-EditX-AutoAWQ-4bit'

# 加载原始模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 准备校准数据（从训练集采样）
# quant_config = { 'zero_point': True, 'q_group_size': 128, 'w_bit': 4 }

# 量化并保存
# model.quantize(tokenizer, quant_config=quant_config)
# model.save_quantized(quant_path)
"
```

**预期提升**：2-3x 加速，显存节省 40-50%

### 方案 2：使用 vLLM 推理引擎（需要大改）

vLLM 支持高性能 AWQ 推理，但需要：
- 重构 `tts.py` 和 `api_server.py`
- 使用 vLLM 的 LLM 类替代原生 transformers
- 适配批处理和流式生成

**预期提升**：3-5x 加速，支持更大 batch size

### 方案 3：使用 Optimum + BetterTransformer

```bash
pip install optimum accelerate

# 在加载时启用优化
from optimum.bettertransformer import BetterTransformer
model = BetterTransformer.transform(model)
```

**预期提升**：1.2-1.5x 加速（对量化模型提升有限）

### 方案 4：暂时禁用 AWQ 模型（临时方案）

在 UI 和 API 中隐藏 AWQ 选项，仅使用 base 模型：

```python
# app.py / api_server.py
# 注释掉 AWQ 模型加载代码
# if awq_model_path and os.path.exists(awq_model_path):
#     ...
```

## 🎯 建议行动

### 短期（立即可做）
1. ✅ **文档更新**：在 README 中说明 AWQ 模型当前性能问题
2. ✅ **UI 提示**：在模型选择处添加性能警告
3. ⚠️ **默认使用 base**：将默认模型设为 base，AWQ 标记为"实验性"

### 中期（1-2 周）
1. 🔄 **重新量化**：使用 AutoAWQ 重新量化模型并测试
2. 📊 **性能对比**：对比 AutoAWQ vs compressed-tensors
3. 📝 **更新文档**：提供性能基准和选择指南

### 长期（1-2 月）
1. 🚀 **集成 vLLM**：作为可选推理后端
2. ⚡ **批处理优化**：支持批量请求处理
3. 🔬 **A/B 测试**：不同量化方案的音质对比

## 📈 性能基准（供参考）

基于 NVIDIA L40S GPU 的测试结果：

| 模型配置 | 推理时间 (50 tokens) | 吞吐量 (tokens/s) | 显存占用 |
|----------|---------------------|------------------|----------|
| Base (BFloat16) | 1.486s | ~33.6 | 23-24 GB |
| AWQ (compressed-tensors) | 5.861s | ~8.5 | 23-24 GB |
| **理论 AutoAWQ** | ~0.7-1.0s | ~50-70 | 12-15 GB |

## 🔗 相关资源

- [AutoAWQ GitHub](https://github.com/casper-hansen/AutoAWQ)
- [compressed-tensors 文档](https://github.com/vllm-project/llm-compressor)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [Transformers 量化指南](https://huggingface.co/docs/transformers/quantization)

## 📌 结论

当前 AWQ 模型使用 compressed-tensors 格式，缺少高性能推理支持，导致：
- ✅ 磁盘空间节省 56%（从 16GB → 7.1GB）
- ❌ 推理速度慢 3.94 倍
- ❌ 显存占用未明显降低

**建议暂时使用 base 模型**，待使用 AutoAWQ 重新量化后再启用量化版本。
