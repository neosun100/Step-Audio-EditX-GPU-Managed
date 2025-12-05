# 性能优化脚本

本目录包含 FunASR 音频编码器的性能优化工具。

## 📁 文件说明

### `quick_optimize.py`
快速优化测试脚本，用于：
- 基准测试当前编码器性能
- 测试轻量级模型（paraformer-base）
- 对比优化前后的性能差异

**使用方法**：

```bash
# 基础测试（只测当前模型）
python scripts/quick_optimize.py

# 完整测试（包括轻量模型对比）
python scripts/quick_optimize.py --test-lightweight

# 指定测试音频
python scripts/quick_optimize.py \
  --test-audio /path/to/your/audio.wav \
  --benchmark-runs 5
```

**参数说明**：
- `--encoder-path`: 音频编码器路径（默认：`/model/Step-Audio-Tokenizer`）
- `--test-audio`: 测试音频文件（默认：`/app/examples/zero_shot_en_prompt.wav`）
- `--test-lightweight`: 是否测试轻量级模型
- `--benchmark-runs`: 基准测试运行次数（默认：3）

**输出示例**：

```
🎯 FunASR 快速优化测试
═══════════════════════════════════════════════════════════
🚀 启用 TF32 加速...
   ✅ TF32 已启用

📦 加载当前模型...
📊 基准测试编码器（3 次运行）...
   预热中...
   运行 1/3: 18.23s
   运行 2/3: 18.15s
   运行 3/3: 18.19s

   ✅ 平均耗时: 18.19s

═══════════════════════════════════════════════════════════
📊 基线性能（paraformer-large + TF32）
═══════════════════════════════════════════════════════════
平均耗时: 18.19s

🧪 测试轻量级模型: paraformer-base...
📊 基准测试编码器（3 次运行）...
   运行 1/3: 7.83s
   运行 2/3: 7.79s
   运行 3/3: 7.81s

   ✅ 平均耗时: 7.81s

═══════════════════════════════════════════════════════════
📊 轻量模型性能（paraformer-base + TF32）
═══════════════════════════════════════════════════════════
平均耗时: 7.81s
提速: 2.33x

✅ 建议切换到 paraformer-base！
   预期总流程提速: 43%
```

---

## 🚀 快速开始

### 1. 立即应用优化（推荐）

```bash
# 进入项目目录
cd /home/neo/upload/Step-Audio-EditX

# 应用优化补丁
patch -p1 < patches/tokenizer_quick_optimize.patch

# 测试效果
python scripts/quick_optimize.py --test-lightweight
```

### 2. 重新构建并部署

```bash
# 重新构建 Docker 镜像
docker build -t step-audio-editx:optimized .

# 停止旧容器
docker stop step-audio-ui step-audio-api

# 启动优化后的容器
docker run -d --name step-audio-ui-opt \
  --gpus '"device=0"' \
  -p 7860:7860 \
  -v /model:/model \
  step-audio-editx:optimized \
  python app.py \
    --model-path /model/Step-Audio-EditX \
    --bnb-model-path /model/Step-Audio-EditX-bnb-4bit \
    --enable-auto-transcribe
```

---

## 📚 相关文档

- **完整优化指南**: [`../docs/funasr-optimization-guide.md`](../docs/funasr-optimization-guide.md)
- **UI 实测结果**: [`../docs/ui-performance-test-result.md`](../docs/ui-performance-test-result.md)
- **性能深度分析**: [`../docs/performance-deep-dive.md`](../docs/performance-deep-dive.md)

---

## 💡 优化方案总结

| 方案 | 预期提速 | 实施难度 | 推荐度 |
|------|---------|---------|--------|
| **启用 TF32** | 1.1-1.2x | 极低 ⭐ | ⭐⭐⭐⭐⭐ |
| **轻量模型** | 2-4x | 低 ⭐ | ⭐⭐⭐⭐⭐ |
| **Torch Compile** | 1.5-2x | 低 ⭐ | ⭐⭐⭐⭐ |
| **ONNX + TensorRT** | 2-3x | 中 ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **批处理** | 1.5-2x | 中 ⭐⭐ | ⭐⭐⭐⭐ |

**最佳实践**：先应用 TF32 + 轻量模型（5 分钟），即可获得 2-3x 提速！
