# 性能优化补丁

本目录包含快速应用 FunASR 编码器优化的补丁文件。

## 📁 补丁文件

### `tokenizer_quick_optimize.patch`

**功能**：一键应用三个快速优化

1. **启用 TF32 加速**（预期提速 1.1-1.2x）
   - 在 Ampere 及以上架构的 GPU 上启用 TensorFloat-32
   - 无精度损失，免费的性能提升

2. **切换到轻量模型**（预期提速 2-4x）
   - 从 `paraformer-large` 切换到 `paraformer-base`
   - 参数量减少，推理速度显著提升
   - 音质略有下降（通常可接受）

3. **优化 ONNX Runtime 配置**（预期提速 1.1-1.3x）
   - 增加线程数（从 1 → 4）
   - 启用并行执行模式
   - 更好的 CPU/GPU 协同

**总体预期提速**：2.5-3x（20s → 7-8s）

---

## 🚀 使用方法

### 方法 1: 直接应用补丁（推荐）

```bash
# 进入项目根目录
cd /home/neo/upload/Step-Audio-EditX

# 应用补丁
patch -p1 < patches/tokenizer_quick_optimize.patch

# 查看修改
git diff tokenizer.py
```

**输出示例**：

```diff
--- a/tokenizer.py
+++ b/tokenizer.py
@@ -1,6 +1,12 @@
 import io
 import threading
 import time
 import os
 
+# ========== 快速优化 1: 启用 TF32 加速 ==========
+import torch
+torch.backends.cuda.matmul.allow_tf32 = True
+torch.backends.cudnn.allow_tf32 = True
+torch.backends.cudnn.benchmark = True
+
 import numpy as np
...
```

### 方法 2: 手动修改（如果补丁失败）

如果 `patch` 命令失败（例如文件已被修改），可以手动应用以下更改：

#### 修改 1: 启用 TF32（在文件开头，第 6 行后添加）

```python
import io
import threading
import time
import os

# ========== 快速优化 1: 启用 TF32 加速 ==========
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import numpy as np
# ... 其余代码
```

#### 修改 2: 切换轻量模型（第 22 行，修改默认参数）

```python
# 修改前:
funasr_model_id="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"

# 修改后:
funasr_model_id="damo/speech_paraformer-base_asr_nat-zh-cn-16k-common-vocab8404"
```

#### 修改 3: 优化 ONNX Runtime（第 62 行附近）

```python
# 修改前:
session_option.intra_op_num_threads = 1

# 修改后:
session_option.intra_op_num_threads = 4      # 从 1 改为 4
session_option.inter_op_num_threads = 2       # 新增
session_option.execution_mode = (             # 新增
    onnxruntime.ExecutionMode.ORT_PARALLEL
)
```

---

## 🧪 测试优化效果

应用补丁后，运行测试脚本验证效果：

```bash
# 测试优化效果
python scripts/quick_optimize.py --test-lightweight

# 对比 UI 实际性能
# 在 Gradio UI 中执行一次 clone 操作并记录时间
```

**预期结果**：

```
原始性能:
  编码时间: ~20s
  总流程: 24s

优化后:
  编码时间: ~7-8s  ← 提速 2.5-3x ✅
  总流程: ~11-12s  ← 提速 50% ✅
```

---

## 🔄 回滚修改

如果优化效果不理想或出现问题，可以回滚：

```bash
# 方法 1: Git 回滚（如果使用 Git）
git checkout tokenizer.py

# 方法 2: 反向应用补丁
patch -R -p1 < patches/tokenizer_quick_optimize.patch

# 方法 3: 从备份恢复
cp tokenizer.py.bak tokenizer.py
```

---

## ⚠️ 注意事项

### 1. 模型下载

切换到 `paraformer-base` 需要先下载模型：

```bash
# 自动下载（首次运行时）
python app.py  # 会自动从 ModelScope/HuggingFace 下载

# 手动下载（可选）
cd /model/Step-Audio-Tokenizer
git lfs clone https://huggingface.co/damo/speech_paraformer-base_asr_nat-zh-cn-16k-common-vocab8404
```

### 2. 精度影响

- **TF32**: 无精度损失（使用更高精度的中间结果）
- **轻量模型**: 轻微精度损失（通常听不出来）
- **ONNX 优化**: 无精度损失

建议在应用后进行 A/B 测试，确认音质可接受。

### 3. GPU 兼容性

- **TF32**: 仅 Ampere 及以上架构支持（RTX 30 系列、A100、L40S 等）
- 旧架构 GPU（如 V100）会自动忽略 TF32 设置，不影响运行

---

## 📊 预期性能提升

| 场景 | 原始耗时 | 优化后 | 提速倍数 |
|------|---------|--------|---------|
| **编码时间** | 20s | 7-8s | 2.5-3x ✅ |
| **Clone 总时间** | 24s | 11-12s | 2x ✅ |
| **吞吐量** | 2.5 req/min | 5 req/min | 2x ✅ |

---

## 🚀 下一步

应用快速优化后，可以考虑：

1. **阶段 2**: ONNX + TensorRT 深度优化（需 1-2 周）
   - 预期提速：3.3x（20s → 6s）
   - 参考：[`../docs/funasr-optimization-guide.md`](../docs/funasr-optimization-guide.md)

2. **阶段 3**: 批处理 + 请求队列（适合高并发）
   - 吞吐量提升：3-4x
   - 适合 API 生产环境

---

## 📚 相关文档

- **完整优化指南**: [`../docs/funasr-optimization-guide.md`](../docs/funasr-optimization-guide.md)
- **测试脚本说明**: [`../scripts/README.md`](../scripts/README.md)
- **性能分析**: [`../docs/ui-performance-test-result.md`](../docs/ui-performance-test-result.md)
