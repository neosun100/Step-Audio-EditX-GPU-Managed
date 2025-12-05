# 🔍 Docker 部署完整审计报告

生成时间：2025-11-22 00:52

---

## 📊 当前状态总结

### ✅ 已运行容器

| 容器名称 | 状态 | 端口映射 | 重启策略 | GPU | 问题 |
|---------|------|---------|---------|-----|------|
| **step-audio-ui-opt** | ✅ 运行中 | 7860→7860 | **❌ no** | 未指定 | **缺少自动重启** |

### ❌ 缺失容器

| 容器名称 | 应该状态 | 应该端口 | 用途 |
|---------|---------|---------|------|
| **step-audio-api** | 运行中 | 8000 | OpenAI兼容API服务 |

### 🚨 关键发现

1. **UI 容器缺少自动重启策略**  
   - 当前：`RestartPolicy=no`
   - 问题：服务器重启后不会自动启动
   - 影响：需要手动重启容器

2. **API 容器完全缺失**  
   - 状态：未创建/未运行
   - 问题：无法通过API访问服务
   - 影响：https://step-api.aws.xin 无法工作

3. **端口冲突风险**  
   - 8000端口已被 `olmocr2-webui` 占用
   - 8001端口已被 `deepseek-ocr-webui-original` 占用
   - API容器需要使用其他端口（建议：8003）

---

## 🔧 问题详细分析

### 问题 1：UI 容器重启策略

#### 当前状态
```bash
$ docker inspect step-audio-ui-opt --format '{{.HostConfig.RestartPolicy.Name}}'
no
```

#### 期望状态
```bash
RestartPolicy=always  # 或 unless-stopped
```

#### 影响
- ❌ 服务器重启后容器不会自动启动
- ❌ 容器异常退出后不会自动重启
- ❌ 需要手动运行 `docker start`

---

### 问题 2：API 容器缺失

#### 当前状态
- 容器不存在
- 无法通过 API 访问服务
- 文档中提到的 API 功能无法使用

#### 应该状态
- API 容器运行在独立 GPU
- 提供 OpenAI 兼容 API
- 支持所有 UI 功能（包括缓存优化）

#### 功能差异检查

根据 `api_server.py` 和 `app.py` 对比：

| 功能 | UI容器 | API容器(应该) | 状态 |
|-----|--------|--------------|------|
| **FunASR 缓存** | ✅ 已实现 | ❓ 需确认 | 🔴 |
| **缓存持久化** | ✅ 已实现 | ❓ 需确认 | 🔴 |
| **Model Variant** | ✅ base/awq/bnb | ✅ 支持 | ✅ |
| **Intensity 参数** | ✅ 0.1-3.0 | ✅ 支持 | ✅ |
| **TF32 加速** | ✅ 已启用 | ❓ 需确认 | 🔴 |
| **ONNX 优化** | ✅ 已优化 | ❓ 需确认 | 🔴 |
| **Whisper ASR** | ✅ 已启用 | ✅ 支持 | ✅ |

---

## ✅ 完整解决方案

### 方案 1：修复 UI 容器重启策略

#### 步骤 1：停止现有容器
```bash
docker stop step-audio-ui-opt
```

#### 步骤 2：更新重启策略
```bash
docker update --restart=always step-audio-ui-opt
```

#### 步骤 3：启动容器
```bash
docker start step-audio-ui-opt
```

#### 步骤 4：验证
```bash
docker inspect step-audio-ui-opt --format '{{.HostConfig.RestartPolicy.Name}}'
# 应该输出: always
```

---

### 方案 2：创建 API 容器（完整优化版）

#### 端口规划
- UI 容器：7860
- API 容器：**8003**（避免与8000/8001冲突）

#### GPU 分配
- UI 容器：GPU 2（当前使用）
- API 容器：GPU 1（推荐，10.5GB 可用）

#### 完整启动命令

```bash
docker run -d \
  --name step-audio-api-opt \
  --restart=always \
  --gpus '"device=1"' \
  -p 8003:8000 \
  -v /home/neo/upload/Step-Audio-EditX:/app \
  -v /home/neo/upload/Step-Audio-EditX/models:/app/models:ro \
  -v /home/neo/upload/Step-Audio-EditX/cache:/app/cache \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e OMP_NUM_THREADS=8 \
  -e MKL_NUM_THREADS=8 \
  step-audio-editx:latest \
  python api_server.py \
    --model-path /app/models/Step-Audio-EditX \
    --tokenizer-model-id /app/models/Step-Audio-Tokenizer \
    --model-source local \
    --enable-auto-transcribe \
    --api-host 0.0.0.0 \
    --api-port 8000
```

#### 参数说明

| 参数 | 值 | 说明 |
|-----|---|------|
| `--name` | step-audio-api-opt | 容器名称 |
| `--restart` | **always** | ✅ 自动重启 |
| `--gpus` | "device=1" | 使用 GPU 1 |
| `-p` | 8003:8000 | 宿主机8003 → 容器8000 |
| `-v` | /app | 代码目录（读写） |
| `-v` | /app/models:ro | 模型目录（只读） |
| `-v` | /app/cache | ✅ 缓存持久化 |
| `CUDA_VISIBLE_DEVICES` | 0 | 容器内GPU编号 |
| `--enable-auto-transcribe` | | ✅ Whisper 转写 |

---

## 🔍 API 容器优化检查清单

### 需要验证的优化项

1. **FunASR 缓存是否启用？**
   - 位置：`tokenizer.py` 中的 `wav2token()` 方法
   - 状态：✅ 已在 UI 容器中实现
   - API 容器：需要确认是否使用相同代码

2. **TF32 加速是否启用？**
   - 位置：`model_loader.py` 或主启动文件
   - 状态：需要添加
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   ```

3. **ONNX Runtime 优化？**
   - 位置：`tokenizer.py` FunASR 初始化
   - 状态：需要确认线程数设置

4. **缓存目录挂载？**
   - 位置：`-v /home/neo/upload/Step-Audio-EditX/cache:/app/cache`
   - 状态：✅ 已在启动命令中添加

---

## 📋 执行计划

### 第一步：修复 UI 容器自动重启

```bash
# 1. 更新重启策略（无需停止容器）
docker update --restart=always step-audio-ui-opt

# 2. 验证
docker inspect step-audio-ui-opt --format '{{.HostConfig.RestartPolicy.Name}}'
```

### 第二步：检查 API 代码优化状态

```bash
# 1. 检查 tokenizer.py 是否有缓存逻辑
grep -n "Cache HIT\|Cache MISS" /home/neo/upload/Step-Audio-EditX/tokenizer.py

# 2. 检查 api_server.py 是否启用 TF32
grep -n "allow_tf32\|TF32" /home/neo/upload/Step-Audio-EditX/api_server.py

# 3. 检查 model_loader.py 优化
grep -n "allow_tf32\|OMP_NUM_THREADS" /home/neo/upload/Step-Audio-EditX/model_loader.py
```

### 第三步：创建 API 容器

```bash
# 1. 构建镜像（如果需要）
cd /home/neo/upload/Step-Audio-EditX
# docker build -t step-audio-editx:latest .  # 如果镜像已存在，跳过

# 2. 启动 API 容器
docker run -d \
  --name step-audio-api-opt \
  --restart=always \
  --gpus '"device=1"' \
  -p 8003:8000 \
  -v /home/neo/upload/Step-Audio-EditX:/app \
  -v /home/neo/upload/Step-Audio-EditX/models:/app/models:ro \
  -v /home/neo/upload/Step-Audio-EditX/cache:/app/cache \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e OMP_NUM_THREADS=8 \
  -e MKL_NUM_THREADS=8 \
  step-audio-editx:latest \
  python api_server.py \
    --model-path /app/models/Step-Audio-EditX \
    --tokenizer-model-id /app/models/Step-Audio-Tokenizer \
    --model-source local \
    --enable-auto-transcribe \
    --api-host 0.0.0.0 \
    --api-port 8000

# 3. 等待启动
sleep 180

# 4. 检查日志
docker logs step-audio-api-opt 2>&1 | tail -30

# 5. 测试健康检查
curl http://localhost:8003/healthz
```

### 第四步：验证两个容器一致性

```bash
# 1. 检查缓存统计（UI）
# 通过浏览器访问 UI，执行 CLONE，查看缓存统计

# 2. 检查缓存统计（API）
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "step-audio-editx",
    "input": "测试文本",
    "step_audio": {
      "mode": "clone",
      "prompt_text": "测试",
      "prompt_audio_url": "https://example.com/test.wav"
    }
  }'

# 3. 检查 Docker 日志中的缓存信息
docker logs step-audio-api-opt 2>&1 | grep -E "Cache (HIT|MISS)"
```

---

## 🎯 最终状态目标

### 容器配置对比

| 项目 | UI 容器 | API 容器 |
|-----|--------|---------|
| **名称** | step-audio-ui-opt | step-audio-api-opt |
| **重启策略** | ✅ always | ✅ always |
| **GPU** | GPU 2 | GPU 1 |
| **端口** | 7860→7860 | 8003→8000 |
| **代码挂载** | ✅ /app | ✅ /app |
| **模型挂载** | ✅ :ro | ✅ :ro |
| **缓存挂载** | ✅ /app/cache | ✅ /app/cache |
| **FunASR缓存** | ✅ 已启用 | 🔴 需确认 |
| **TF32加速** | ✅ 已启用 | 🔴 需添加 |
| **ONNX优化** | ✅ 已优化 | 🔴 需确认 |
| **Whisper** | ✅ 已启用 | ✅ 已启用 |
| **Model Variant** | ✅ base/awq/bnb | ✅ 支持 |
| **Intensity** | ✅ 0.1-3.0 | ✅ 支持 |

---

## 🔍 需要检查的代码文件

### 1. api_server.py
检查项：
- [ ] 是否导入并使用了优化后的 `tokenizer.py`？
- [ ] 是否启用 TF32？
- [ ] 是否正确初始化缓存？

### 2. tokenizer.py
检查项：
- [x] `wav2token()` 是否有缓存逻辑？✅
- [x] 是否正确更新 `cache_hits` 和 `cache_misses`？✅
- [x] 是否支持持久化缓存？✅

### 3. model_loader.py
检查项：
- [ ] 是否启用 TF32？
- [ ] ONNX Runtime 配置是否优化？

---

## ⚠️ 风险提示

1. **GPU 内存**  
   - 每个容器需要 ~28GB GPU 显存
   - GPU 1 当前可用：10.5GB
   - 建议：清理 GPU 1 上的其他进程

2. **端口冲突**  
   - 8000 已被 olmocr2 占用
   - 解决：API 容器使用 8003

3. **代码同步**  
   - UI 和 API 容器都挂载 `/app`
   - 任一修改会影响两者
   - 需要确保代码一致性

---

## 📝 检查清单

### UI 容器
- [ ] 重启策略改为 `always`
- [x] FunASR 缓存已启用
- [x] 缓存持久化已配置
- [x] TF32 加速已启用
- [x] Model Variant 支持（base/awq/bnb）
- [x] Intensity 参数支持（0.1-3.0）

### API 容器
- [ ] 创建容器
- [ ] 重启策略设为 `always`
- [ ] FunASR 缓存确认启用
- [ ] 缓存持久化配置
- [ ] TF32 加速确认/添加
- [ ] Model Variant 支持确认
- [ ] Intensity 参数支持确认
- [ ] 端口映射正确（8003→8000）
- [ ] GPU 分配正确（GPU 1）

### 一致性验证
- [ ] 两个容器使用相同代码
- [ ] 两个容器有相同优化
- [ ] 两个容器缓存目录共享
- [ ] 测试 UI 和 API 缓存都工作
- [ ] 性能对比（应该相同）

---

## 🚀 下一步行动

1. **立即执行**：修复 UI 容器重启策略
2. **检查代码**：确认 API 相关文件的优化状态
3. **创建容器**：启动 API 容器
4. **验证一致性**：测试两个容器功能和性能
5. **更新文档**：记录最终配置

---

**生成时间**：2025-11-22 00:52:00  
**审计人员**：AI Assistant  
**状态**：待执行
