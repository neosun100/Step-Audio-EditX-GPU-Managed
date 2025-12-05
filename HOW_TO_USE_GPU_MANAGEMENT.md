# 🎮 如何使用 GPU 显存智能管理

## 🚀 3 步开始使用

### 步骤 1：编辑启动脚本

```bash
vim start_with_gpu_management.sh
```

修改这 3 个配置：
```bash
PROJECT_DIR="/path/to/Step-Audio-EditX"  # 改为你的项目路径
PORT=7860                                  # 服务端口（如果冲突就改）
GPU_IDLE_TIMEOUT=600                       # 空闲超时（秒），默认 10 分钟
```

### 步骤 2：启动服务

```bash
./start_with_gpu_management.sh
```

等待 3 分钟，看到这个信息表示成功：
```
✅ 服务启动成功！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌐 访问地址：
   UI 界面: http://0.0.0.0:7860
```

### 步骤 3：使用 UI

1. 打开浏览器：http://localhost:7860
2. 找到 **🎮 GPU 显存管理** 面板
3. 点击 **🔄 刷新状态** 查看 GPU 状态

---

## 📊 效果验证

### 验证 1：查看初始显存

```bash
nvidia-smi
```

应该看到显存占用很低（< 1GB），因为模型还未加载。

### 验证 2：执行一次任务

在 UI 中：
1. 上传音频文件
2. 输入文本
3. 点击 **CLONE** 按钮

### 验证 3：观察显存变化

```bash
watch -n 1 nvidia-smi
```

你会看到：
- **任务开始**：显存上升到 ~40GB
- **任务完成**：显存下降到 ~1GB（2秒内）✅

### 验证 4：查看 GPU 状态

在 UI 的 **🎮 GPU 显存管理** 面板中：
- 点击 **🔄 刷新状态**
- 应该看到：`• tts: 🟡 CPU`

---

## 🎯 核心功能

### 功能 1：懒加载

**现象**：
- 启动后不占用显存
- 首次请求时加载（20-30秒）

**验证**：
```bash
# 启动后立即查看
nvidia-smi
# 显存应该很低（< 1GB）
```

### 功能 2：即用即卸

**现象**：
- 任务完成后 2 秒内释放显存
- 显存从 40GB 降至 1GB

**验证**：
```bash
# 执行任务时观察
watch -n 1 nvidia-smi
# 任务完成后显存应该快速下降
```

### 功能 3：快速恢复

**现象**：
- 第二次请求比首次快
- 从 CPU 恢复只需 2-5秒

**验证**：
- 第一次任务：~24秒
- 第二次任务：~26-29秒（+2-5秒）

### 功能 4：自动监控

**现象**：
- 空闲 10 分钟后自动卸载
- 无需手动管理

**验证**：
```bash
# 执行任务后等待 10 分钟
# 然后查看显存
nvidia-smi
# 应该保持在 ~1GB
```

### 功能 5：手动控制

**现象**：
- UI 提供控制按钮
- 可以手动卸载/释放

**验证**：
1. 点击 **💾 卸载到CPU**
2. 运行 `nvidia-smi`
3. 显存应该下降

---

## 🎮 UI 控制面板

### 面板位置

在 UI 左侧，找到 **🎮 GPU 显存管理** 面板（在 FunASR 缓存统计下方）。

### 显示内容

```
🎮 GPU 显存管理状态
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU 显存占用: 1024.5 MB
空闲超时: 600 秒

📦 模型状态:
  • tts: 🟡 CPU
    空闲时间: 45 秒
```

### 按钮功能

| 按钮 | 功能 | 何时使用 |
|------|------|---------|
| 🔄 刷新状态 | 查看最新状态 | 随时 |
| 💾 卸载到CPU | 立即释放显存 | 任务完成后 |
| 🗑️ 完全释放 | 清空所有缓存 | 长期不用时 |

### 状态图标

- 🟢 GPU：模型在 GPU 上（正在使用）
- 🟡 CPU：模型在 CPU 上（已卸载）
- ⚪ 未加载：模型未加载

---

## 💡 使用技巧

### 技巧 1：根据使用频率调整超时

```bash
# 高频使用（每分钟多次）：不启用 GPU 管理
python app.py --model-path /path/to/models

# 中频使用（每分钟 1-10 次）：10 分钟超时
python app.py \
  --model-path /path/to/models \
  --enable-gpu-management \
  --gpu-idle-timeout 600

# 低频使用（每小时几次）：30 分钟超时
python app.py \
  --model-path /path/to/models \
  --enable-gpu-management \
  --gpu-idle-timeout 1800
```

### 技巧 2：多服务共享 GPU

```bash
# 服务 A（端口 7860）
./start_with_gpu_management.sh  # GPU_IDLE_TIMEOUT=300

# 服务 B（端口 7861）
# 修改 PORT=7861，然后启动
./start_with_gpu_management.sh
```

**效果**：
- 两个服务共享同一张 GPU
- 空闲时各占 1GB，总共 2GB
- 使用时轮流占用 40GB

### 技巧 3：配合缓存使用

启动时同时启用 GPU 管理和自动转写：
```bash
python app.py \
  --model-path /path/to/models \
  --enable-gpu-management \
  --gpu-idle-timeout 600 \
  --enable-auto-transcribe
```

**综合效果**：
- FunASR 缓存：24s → 8s
- GPU 管理：40GB → 1GB
- 性能 + 资源双提升

---

## 🔧 常见问题

### Q1：GPU 管理未生效？

**检查**：
```bash
docker logs step-audio-gpu-managed | grep "GPU 管理"
```

**应该看到**：`🚀 GPU 管理已启用`

**如果没有**：
- 检查启动脚本中的配置
- 确认使用了 `--enable-gpu-management` 参数

### Q2：UI 没有 GPU 控制面板？

**原因**：GPU 管理未启用

**解决**：
- 确认启动参数包含 `--enable-gpu-management`
- 重启服务

### Q3：显存没有释放？

**检查**：
1. 确认任务已完成
2. 等待 2-3 秒
3. 运行 `nvidia-smi` 查看

**如果仍未释放**：
- 点击 UI 中的 **💾 卸载到CPU** 按钮
- 或点击 **🗑️ 完全释放** 按钮

### Q4：恢复速度很慢？

**可能原因**：
- CPU 内存不足
- 系统负载过高

**检查**：
```bash
# 检查内存
free -h

# 检查负载
top
```

---

## 📚 更多文档

### 快速参考
- [快速开始](QUICK_START_GPU.md) - 3 分钟上手
- [本文档](HOW_TO_USE_GPU_MANAGEMENT.md) - 使用说明

### 详细文档
- [完整指南](GPU_MANAGEMENT.md) - 详细使用说明
- [功能总结](GPU_MANAGEMENT_SUMMARY.md) - 技术实现
- [改动说明](GPU_MANAGEMENT_CHANGES.md) - 代码改动

### 测试
- [测试脚本](test_gpu_management.py) - 功能测试

---

## 🎉 开始使用

现在你已经了解了如何使用 GPU 显存管理功能！

**记住 3 个关键点**：
1. ✅ 编辑启动脚本配置
2. ✅ 运行 `./start_with_gpu_management.sh`
3. ✅ 在 UI 中查看 GPU 状态

**享受智能显存管理带来的便利！** 🚀
