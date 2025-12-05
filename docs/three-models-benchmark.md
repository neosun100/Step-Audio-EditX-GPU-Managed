# Step-Audio-EditX 三模型性能对比报告

## 📊 基准测试结果（NVIDIA L40S）

| 模型变体 | 加载时间 | 推理时间 (50 tokens) | 相对速度 | 模型大小 | 量化方法 |
|----------|---------|---------------------|---------|----------|----------|
| **Base** | 2.34s | **1.636s** | 1.00x ⚡ | 16 GB | BFloat16 (无量化) |
| **BnB 4-bit** | 1.98s | 3.532s | 0.46x | 7.1 GB | BitsAndBytes NF4 |
| **AWQ 4-bit** | 2.80s | 5.942s | 0.28x | 7.1 GB | compressed-tensors |

## 🏆 性能排名

1. **Base 模型** - 最快（1.636s）✅ **强烈推荐用于生产环境**
2. **BnB 4-bit 模型** - 中等（3.532s，比 base 慢 2.16x）⚠️ **实验性**
3. **AWQ 4-bit 模型** - 最慢（5.942s，比 base 慢 3.63x）❌ **不推荐**

## 📈 详细分析

### Base 模型（推荐）
```yaml
优势:
  - 推理速度最快（1.636s）
  - 音质最佳（无量化损失）
  - 最稳定、最成熟
  - 无额外依赖

劣势:
  - 磁盘占用最大（16GB）
  - 显存占用较高（~24GB）

适用场景:
  - 生产环境
  - 对延迟敏感的应用
  - 追求最佳音质
```

### BnB 4-bit 模型（实验性）
```yaml
优势:
  - 磁盘空间节省 56%（7.1GB vs 16GB）
  - 加载时间最快（1.98s）
  - 使用成熟的 BitsAndBytes 量化
  - 比 AWQ 快 68%

劣势:
  - 推理慢 2.16 倍
  - 音质可能有轻微下降
  - 显存占用未明显降低（~24GB）

适用场景:
  - 磁盘空间受限
  - 可接受 2x 延迟增加
  - 测试量化效果
```

### AWQ 4-bit 模型（不推荐）
```yaml
优势:
  - 磁盘空间节省 56%

劣势:
  - 推理慢 3.63 倍（最慢）
  - 使用 compressed-tensors 格式（缺少优化）
  - 无实际生产价值

结论:
  建议暂时不使用，等待 AutoAWQ 格式优化版本
```

## 💡 推荐使用指南

### 场景 1：生产环境
```bash
# 使用 Base 模型
"step_audio": {
  "model_variant": "base"  # 默认，最快
}
```

### 场景 2：测试 BnB 量化
```bash
# 使用 BnB 模型
"step_audio": {
  "model_variant": "bnb"  # 慢 2.16x，节省 56% 磁盘
}
```

### 场景 3：开发调试
```bash
# Base 模型提供最快反馈
"step_audio": {
  "model_variant": "base"
}
```

## 🔧 量化技术对比

### BitsAndBytes NF4 (bnb 模型)
- **类型**: 动态量化，推理时反量化
- **优势**: 成熟稳定，社区广泛使用
- **劣势**: 每次前向传播都需要反量化
- **性能**: 比 base 慢 2.16x

### compressed-tensors (awq 模型)
- **类型**: 静态量化，权重压缩
- **优势**: 磁盘占用小
- **劣势**: 缺少优化的 CUDA kernel，Python 实现慢
- **性能**: 比 base 慢 3.63x ❌

### 理想的 AutoAWQ (未来)
- **类型**: 硬件优化的融合 kernel
- **预期性能**: 比 base 快 1.5-2x ⚡
- **预期显存**: 节省 40-50%
- **状态**: 等待官方或社区提供

## 🎯 开发路线图

### 短期（当前）
- ✅ 支持三种模型变体（base/bnb/awq）
- ✅ 提供性能基准测试
- ✅ 文档完善

### 中期（1-2 周）
- [ ] 优化 BnB 模型加载流程
- [ ] 测试音质对比
- [ ] 添加 A/B 测试工具

### 长期（1-2 月）
- [ ] 集成 AutoAWQ 格式（如果官方发布）
- [ ] 支持 vLLM 推理后端
- [ ] 实现批处理优化

## 📝 使用建议

**优先级排序**：
1. **Base 模型** - 默认首选，性能最佳
2. **BnB 模型** - 磁盘受限时可尝试
3. **AWQ 模型** - 不推荐使用

**关键提示**：
- 三个模型音质理论上相同（都基于同一个训练权重）
- 性能差异主要来自量化方法的推理开销
- 显存占用三者接近（~24GB），因为 CosyVoice 占主要部分

## 🚀 快速开始

### UI 使用
1. 访问 `http://<IP>:7860`
2. 在右侧 "Model Variant" 选择模型
3. 默认使用 `base`（推荐）

### API 使用
```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "step-audio-editx",
    "voice": "story_teller",
    "input": "测试语音合成效果",
    "step_audio": {
      "mode": "clone",
      "model_variant": "base"  # base | bnb | awq
    }
  }'
```

## 📚 相关文档
- [性能分析详解](model-performance-analysis.md)
- [API 使用指南](api-guide.md)
- [主文档](../README.md)
