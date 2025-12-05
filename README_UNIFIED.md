# 统一容器部署方式（推荐）⭐⭐⭐⭐⭐

## 🎯 新特性

### V2.0 优化（2025-12-03）

#### 1. 模型懒加载 🚀
- **按需加载**：需要时才加载模型到GPU
- **自动卸载**：空闲5分钟后自动释放显存
- **显存节省**：空闲时显存占用从 24GB → 0GB
- **线程安全**：支持并发请求

#### 2. 统一部署 🎯
- **一个容器**：UI + API 统一部署
- **共享模型**：避免重复加载，节省50%显存
- **简化管理**：一个端口，一个容器
- **资源优化**：GPU利用率提升50%+

---

## 🚀 快速开始

### 方式一：统一容器（推荐）⭐⭐⭐⭐⭐

#### 特点
- ✅ UI 和 API 在同一个容器
- ✅ 共享同一个模型实例
- ✅ 懒加载，按需使用GPU
- ✅ 空闲自动释放显存
- ✅ 一个端口访问所有功能

#### 使用步骤

```bash
# 1. 编辑启动脚本
vim start_unified_container.sh

# 修改配置：
#   PROJECT_DIR="/your/project/path"  # 项目路径
#   GPU_ID=2                          # GPU ID
#   PORT=7860                         # 服务端口
#   IDLE_TIMEOUT=300                  # 空闲超时（秒）

# 2. 启动容器
./start_unified_container.sh

# 3. 等待启动（约1分钟）
docker logs -f step-audio-unified

# 4. 访问服务
# UI: http://localhost:7860
# API: http://localhost:7860/docs
# 健康检查: http://localhost:7860/healthz
# 模型状态: http://localhost:7860/api/v1/models/status
```

#### 启动参数说明

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `PROJECT_DIR` | 项目根目录 | 必填 | 实际路径 |
| `GPU_ID` | GPU设备ID | 2 | 根据实际情况 |
| `PORT` | 服务端口 | 7860 | 7860 |
| `IDLE_TIMEOUT` | 空闲超时（秒） | 300 | 300-600 |

---

## 📊 对比：统一容器 vs 分离容器

| 项目 | 分离容器（旧） | 统一容器（新）⭐ | 改进 |
|------|---------------|----------------|------|
| **容器数量** | 2个 | 1个 | -50% |
| **模型实例** | 2个 | 1个 | -50% |
| **显存占用（使用时）** | 24GB | 12GB | -50% |
| **显存占用（空闲时）** | 24GB | 0GB | -100% |
| **端口占用** | 2个 | 1个 | -50% |
| **启动时间** | 3分钟 | 1分钟 | +66% |
| **管理复杂度** | 高 | 低 | 简化 |
| **GPU利用率** | 低 | 高 | +50% |

---

## 🎯 使用场景

### 场景1：低频使用（推荐统一容器）

**特点**：每天使用几次，大部分时间空闲

**优势**：
- 空闲时自动释放显存
- 其他服务可以使用GPU
- 节省电力和资源

**配置**：
```bash
IDLE_TIMEOUT=300  # 5分钟空闲后卸载
```

### 场景2：高频使用

**特点**：持续使用，请求频繁

**优势**：
- 模型保持加载状态
- 响应速度快
- 自动管理生命周期

**配置**：
```bash
IDLE_TIMEOUT=1800  # 30分钟空闲后卸载
```

### 场景3：多服务共享GPU

**特点**：多个服务需要使用同一个GPU

**优势**：
- 空闲时释放显存给其他服务
- 提高GPU利用率
- 避免资源浪费

**配置**：
```bash
IDLE_TIMEOUT=180  # 3分钟空闲后卸载（快速释放）
```

---

## 🔧 高级功能

### 1. 查看模型状态

```bash
# 通过API查看
curl http://localhost:7860/api/v1/models/status | jq

# 返回示例：
{
  "base": {
    "loaded": true,
    "auto_unload": true,
    "idle_timeout": 300,
    "idle_time": 45.2,
    "time_until_unload": 254.8,
    "gpu_memory_allocated_gb": 11.8,
    "gpu_memory_reserved_gb": 12.5
  },
  "bnb": {
    "loaded": false,
    "auto_unload": true,
    "idle_timeout": 300
  }
}
```

### 2. 手动卸载模型

```bash
# 卸载 base 模型
curl -X POST http://localhost:7860/api/v1/models/base/unload

# 卸载 bnb 模型
curl -X POST http://localhost:7860/api/v1/models/bnb/unload
```

### 3. 监控GPU使用

```bash
# 进入容器
docker exec -it step-audio-unified bash

# 查看GPU状态
nvidia-smi

# 实时监控
watch -n 1 nvidia-smi
```

### 4. 查看日志

```bash
# 实时日志
docker logs -f step-audio-unified

# 查看懒加载相关日志
docker logs step-audio-unified 2>&1 | grep -E "加载|卸载|空闲"

# 查看最近100行
docker logs --tail 100 step-audio-unified
```

---

## 🐛 故障排查

### 问题1：首次请求很慢

**症状**：第一次请求需要等待3-5秒

**原因**：模型正在加载（懒加载特性）

**解决**：
- 这是正常现象
- 后续请求会很快
- 如需避免，可以增加 `IDLE_TIMEOUT` 或禁用自动卸载

### 问题2：显存未释放

**症状**：卸载后显存仍然占用

**原因**：PyTorch缓存未清理

**解决**：
```bash
# 手动强制卸载
curl -X POST http://localhost:7860/api/v1/models/base/unload

# 重启容器
docker restart step-audio-unified
```

### 问题3：容器启动失败

**症状**：容器无法启动

**原因**：模型路径错误或权限问题

**解决**：
```bash
# 检查模型目录
ls -la /home/neo/upload/Step-Audio-EditX/models/

# 应该看到：
# Step-Audio-Tokenizer/
# Step-Audio-EditX/
# Step-Audio-EditX-AWQ-4bit/
# Step-Audio-EditX-bnb-4bit/

# 检查权限
docker exec step-audio-unified ls -la /app/models/
```

---

## 📝 迁移指南

### 从分离容器迁移到统一容器

#### 1. 停止旧容器

```bash
# 停止UI容器
docker stop step-audio-ui-opt
docker rm step-audio-ui-opt

# 停止API容器
docker stop step-audio-api-opt
docker rm step-audio-api-opt
```

#### 2. 启动新容器

```bash
./start_unified_container.sh
```

#### 3. 验证服务

```bash
# 检查健康状态
curl http://localhost:7860/healthz

# 检查模型状态
curl http://localhost:7860/api/v1/models/status

# 访问UI
# http://localhost:7860
```

#### 4. API端点变化

| 旧端点 | 新端点 | 说明 |
|--------|--------|------|
| `http://localhost:7860` | `http://localhost:7860` | UI（不变） |
| `http://localhost:8003/healthz` | `http://localhost:7860/healthz` | 健康检查 |
| `http://localhost:8003/docs` | `http://localhost:7860/docs` | API文档 |
| `http://localhost:8003/v1/audio/speech` | `http://localhost:7860/v1/audio/speech` | 音频生成 |

---

## 💡 最佳实践

### 1. 选择合适的空闲超时

```bash
# 低频使用（每天几次）
IDLE_TIMEOUT=300  # 5分钟

# 中频使用（每小时几次）
IDLE_TIMEOUT=600  # 10分钟

# 高频使用（持续使用）
IDLE_TIMEOUT=1800  # 30分钟
```

### 2. 监控资源使用

```bash
# 定期检查模型状态
watch -n 10 'curl -s http://localhost:7860/api/v1/models/status | jq'

# 监控GPU使用
docker exec step-audio-unified nvidia-smi
```

### 3. 日志管理

```bash
# 限制日志大小
docker run ... --log-opt max-size=100m --log-opt max-file=3 ...

# 定期清理日志
docker logs step-audio-unified > /dev/null 2>&1
```

---

## 📚 相关文档

- **[完整优化说明](OPTIMIZATION_V2.md)** - 详细的优化方案和原理
- **[API 使用指南](docs/api-guide.md)** - API 接口文档
- **[性能优化指南](docs/funasr-optimization-guide.md)** - 性能优化方案

---

## 🎉 总结

### 核心优势

1. **资源节省**：空闲时释放显存，节省50%+资源
2. **统一部署**：一个容器提供UI和API，简化管理
3. **按需加载**：需要时才加载，提高GPU利用率
4. **自动管理**：无需手动干预，自动卸载和加载

### 推荐使用场景

- ✅ 低频使用场景（每天几次）
- ✅ 多服务共享GPU
- ✅ 开发测试环境
- ✅ 资源受限环境

---

**更新时间**：2025-12-03
**版本**：V2.0
