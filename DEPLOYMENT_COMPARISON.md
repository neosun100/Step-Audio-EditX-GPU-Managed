# 部署方式对比

## 📊 三种部署方式

### 方式一：统一容器（推荐）⭐⭐⭐⭐⭐

**特点**：
- UI + API 在同一个容器
- 共享同一个模型实例
- 懒加载，按需使用GPU
- 空闲自动释放显存

**启动**：
```bash
./start_unified_container.sh
```

**访问**：
- UI: http://localhost:7860
- API: http://localhost:7860/docs

**资源占用**：
- 容器数：1个
- 模型实例：1个
- 显存（使用时）：12GB
- 显存（空闲时）：0GB

**适用场景**：
- ✅ 低频使用
- ✅ 多服务共享GPU
- ✅ 资源受限环境
- ✅ 开发测试

---

### 方式二：UI 容器

**特点**：
- 仅提供 Gradio Web 界面
- 模型常驻内存
- 响应速度快

**启动**：
```bash
./start_ui_container.sh
```

**访问**：
- UI: http://localhost:7860

**资源占用**：
- 容器数：1个
- 模型实例：1个
- 显存：12GB（常驻）

**适用场景**：
- ✅ 仅需要UI
- ✅ 高频使用
- ✅ GPU资源充足

---

### 方式三：API 容器

**特点**：
- 仅提供 REST API
- OpenAI 兼容接口
- 模型常驻内存

**启动**：
```bash
./start_api_container.sh
```

**访问**：
- API: http://localhost:8003/docs

**资源占用**：
- 容器数：1个
- 模型实例：1个
- 显存：12GB（常驻）

**适用场景**：
- ✅ 仅需要API
- ✅ 集成到其他系统
- ✅ 高频使用

---

### 方式四：UI + API 分离（不推荐）

**特点**：
- UI 和 API 分别部署
- 两个独立的模型实例
- 资源占用高

**启动**：
```bash
./start_ui_container.sh
./start_api_container.sh
```

**访问**：
- UI: http://localhost:7860
- API: http://localhost:8003/docs

**资源占用**：
- 容器数：2个
- 模型实例：2个
- 显存：24GB（常驻）

**适用场景**：
- ❌ 不推荐（资源浪费）

---

## 📈 性能对比

| 指标 | 统一容器 | UI容器 | API容器 | UI+API分离 |
|------|---------|--------|---------|-----------|
| 容器数 | 1 | 1 | 1 | 2 |
| 模型实例 | 1 | 1 | 1 | 2 |
| 显存（使用） | 12GB | 12GB | 12GB | 24GB |
| 显存（空闲） | 0GB | 12GB | 12GB | 24GB |
| 启动时间 | 1分钟 | 3分钟 | 3分钟 | 3分钟 |
| 首次响应 | +3秒 | 即时 | 即时 | 即时 |
| GPU利用率 | 高 | 低 | 低 | 低 |
| 推荐度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |

---

## 🎯 选择建议

### 选择统一容器，如果：
- ✅ 每天使用次数 < 100次
- ✅ 需要同时使用UI和API
- ✅ GPU需要被多个服务共享
- ✅ 希望节省资源

### 选择UI容器，如果：
- ✅ 仅需要Web界面
- ✅ 使用频率很高
- ✅ GPU资源充足
- ✅ 对响应速度要求高

### 选择API容器，如果：
- ✅ 仅需要API接口
- ✅ 需要集成到其他系统
- ✅ 使用频率很高
- ✅ GPU资源充足

### 不要选择UI+API分离，因为：
- ❌ 资源浪费（双倍显存）
- ❌ 管理复杂（两个容器）
- ❌ 没有额外好处

---

## 💡 迁移建议

### 从 UI+API分离 → 统一容器

**收益**：
- 节省50%显存
- 简化管理
- 提高GPU利用率

**步骤**：
```bash
# 1. 停止旧容器
docker stop step-audio-ui-opt step-audio-api-opt
docker rm step-audio-ui-opt step-audio-api-opt

# 2. 启动新容器
./start_unified_container.sh

# 3. 验证
curl http://localhost:7860/healthz
```

### 从 UI容器 → 统一容器

**收益**：
- 增加API功能
- 空闲时释放显存

**步骤**：
```bash
# 1. 停止旧容器
docker stop step-audio-ui-opt
docker rm step-audio-ui-opt

# 2. 启动新容器
./start_unified_container.sh
```

---

## 📞 获取帮助

- **完整文档**：[README_UNIFIED.md](README_UNIFIED.md)
- **优化说明**：[OPTIMIZATION_V2.md](OPTIMIZATION_V2.md)
- **API文档**：[docs/api-guide.md](docs/api-guide.md)

---

**更新时间**：2025-12-03
