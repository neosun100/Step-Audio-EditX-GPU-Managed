# 快速开始指南

## 🚀 5分钟快速部署

### 步骤1：编辑配置（1分钟）

```bash
vim start_unified_container.sh
```

修改以下配置：

```bash
PROJECT_DIR="/home/neo/upload/Step-Audio-EditX"  # 改成你的项目路径
GPU_ID=2                                          # 改成你的GPU ID
PORT=7860                                         # 服务端口（可选）
IDLE_TIMEOUT=300                                  # 空闲超时（可选）
```

### 步骤2：启动容器（1分钟）

```bash
./start_unified_container.sh
```

### 步骤3：等待启动（1分钟）

```bash
# 查看日志
docker logs -f step-audio-unified

# 看到以下信息表示启动成功：
# "服务器启动: http://0.0.0.0:7860"
```

### 步骤4：访问服务（1分钟）

```bash
# 健康检查
curl http://localhost:7860/healthz

# 访问UI
# 浏览器打开: http://localhost:7860

# 访问API文档
# 浏览器打开: http://localhost:7860/docs
```

### 步骤5：测试功能（1分钟）

```bash
# 查看模型状态
curl http://localhost:7860/api/v1/models/status | jq

# 预期输出：
{
  "base": {
    "loaded": false,  # 初始未加载
    "auto_unload": true,
    "idle_timeout": 300
  }
}
```

---

## 🎯 核心功能

### 1. 懒加载

**特点**：
- 需要时才加载模型
- 空闲5分钟后自动卸载
- 自动释放GPU显存

**效果**：
- 空闲时：0GB显存
- 使用时：12GB显存

### 2. 统一部署

**特点**：
- UI + API 在同一个容器
- 共享同一个模型实例
- 一个端口访问所有功能

**效果**：
- 容器数：2个 → 1个
- 显存占用：24GB → 12GB

### 3. 自动管理

**特点**：
- 自动加载和卸载
- 无需手动干预
- 线程安全

**效果**：
- 简化运维
- 提高GPU利用率

---

## 📊 使用示例

### 示例1：使用UI

1. 访问 http://localhost:7860
2. 上传参考音频
3. 输入参考文本和目标文本
4. 选择模型变体（base/awq/bnb）
5. 点击"生成"
6. 首次使用会加载模型（3-5秒）
7. 获得生成的音频

### 示例2：使用API

```bash
# 生成音频
curl -X POST http://localhost:7860/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "step-audio-editx",
    "voice": "happy_en",
    "input": "Hello! This is a test.",
    "step_audio": {
      "mode": "clone",
      "model_variant": "base"
    }
  }' \
  --output output.wav
```

### 示例3：监控状态

```bash
# 查看模型状态
curl http://localhost:7860/api/v1/models/status | jq

# 查看GPU使用
docker exec step-audio-unified nvidia-smi

# 查看日志
docker logs --tail 100 step-audio-unified
```

---

## 🔧 常用命令

### 容器管理

```bash
# 启动容器
./start_unified_container.sh

# 停止容器
docker stop step-audio-unified

# 重启容器
docker restart step-audio-unified

# 删除容器
docker stop step-audio-unified && docker rm step-audio-unified

# 查看日志
docker logs -f step-audio-unified

# 进入容器
docker exec -it step-audio-unified bash
```

### 模型管理

```bash
# 查看模型状态
curl http://localhost:7860/api/v1/models/status

# 手动卸载模型
curl -X POST http://localhost:7860/api/v1/models/base/unload

# 查看GPU使用
docker exec step-audio-unified nvidia-smi
```

### 健康检查

```bash
# 检查服务状态
curl http://localhost:7860/healthz

# 检查容器状态
docker ps | grep step-audio-unified

# 检查GPU状态
nvidia-smi
```

---

## 🐛 常见问题

### Q1：首次请求很慢？

**A**：这是正常现象，模型正在加载（3-5秒）。后续请求会很快。

### Q2：显存没有释放？

**A**：等待5分钟空闲超时，或手动卸载：
```bash
curl -X POST http://localhost:7860/api/v1/models/base/unload
```

### Q3：容器启动失败？

**A**：检查配置：
```bash
# 检查模型目录
ls -la /home/neo/upload/Step-Audio-EditX/models/

# 查看日志
docker logs step-audio-unified

# 检查GPU
nvidia-smi
```

### Q4：如何禁用自动卸载？

**A**：修改 `server.py` 启动参数：
```bash
python server.py ... --disable-auto-unload
```

### Q5：如何调整空闲超时？

**A**：修改 `start_unified_container.sh` 中的 `IDLE_TIMEOUT` 变量。

---

## 📚 更多文档

- **[OPTIMIZATION_V2.md](OPTIMIZATION_V2.md)** - 完整优化说明
- **[README_UNIFIED.md](README_UNIFIED.md)** - 统一部署指南
- **[DEPLOYMENT_COMPARISON.md](DEPLOYMENT_COMPARISON.md)** - 部署方式对比
- **[COMPLETION_REPORT.md](COMPLETION_REPORT.md)** - 完成报告

---

## 🎉 总结

### 核心优势

1. **资源节省**：空闲时释放显存，节省50%+资源
2. **统一部署**：一个容器提供UI和API
3. **按需加载**：需要时才加载，提高GPU利用率
4. **自动管理**：无需手动干预

### 5分钟完成

1. ✅ 编辑配置（1分钟）
2. ✅ 启动容器（1分钟）
3. ✅ 等待启动（1分钟）
4. ✅ 访问服务（1分钟）
5. ✅ 测试功能（1分钟）

### 开始使用

```bash
./start_unified_container.sh
```

---

**更新时间**：2025-12-03

**版本**：V2.0
