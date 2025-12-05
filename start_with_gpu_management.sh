#!/bin/bash

# ============================================================================
# Step-Audio-EditX 启动脚本 (GPU 显存智能管理版本)
# ============================================================================
# 功能：
# 1. 自动选择显存占用最少的 GPU
# 2. 启用 GPU 显存智能管理（懒加载 + 即用即卸）
# 3. 支持 UI + API 统一部署
# ============================================================================

set -e

# ============================================================================
# 配置区域 - 请根据实际情况修改
# ============================================================================

PROJECT_DIR="/path/to/Step-Audio-EditX"  # 项目根目录路径
CONTAINER_NAME="step-audio-gpu-managed"   # 容器名称
PORT=7860                                  # 服务端口
GPU_IDLE_TIMEOUT=600                       # GPU 空闲超时（秒），默认 10 分钟
ENABLE_API=true                            # 是否启用 API（true/false）

# ============================================================================
# 自动选择最空闲的 GPU
# ============================================================================

echo "🔍 检测可用 GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 错误：未找到 nvidia-smi 命令"
    echo "   请确保已安装 NVIDIA 驱动"
    exit 1
fi

# 选择显存占用最少的 GPU
GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
         sort -t',' -k2 -n | head -1 | cut -d',' -f1)

if [ -z "$GPU_ID" ]; then
    echo "❌ 错误：无法检测到可用 GPU"
    exit 1
fi

GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID)
GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU_ID)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU_ID)

echo "✅ 已选择 GPU $GPU_ID: $GPU_NAME"
echo "   显存使用: ${GPU_MEM_USED}MB / ${GPU_MEM_TOTAL}MB"

# ============================================================================
# 检查配置
# ============================================================================

if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ 错误：项目目录不存在: $PROJECT_DIR"
    echo "   请修改脚本中的 PROJECT_DIR 变量"
    exit 1
fi

if [ ! -d "$PROJECT_DIR/models" ]; then
    echo "❌ 错误：模型目录不存在: $PROJECT_DIR/models"
    echo "   请先下载模型到 models/ 目录"
    exit 1
fi

# ============================================================================
# 检查端口占用
# ============================================================================

if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  警告：端口 $PORT 已被占用"
    echo "   当前占用进程："
    lsof -Pi :$PORT -sTCP:LISTEN
    read -p "是否继续？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ============================================================================
# 停止并删除旧容器
# ============================================================================

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🛑 停止旧容器: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
    docker rm $CONTAINER_NAME >/dev/null 2>&1 || true
fi

# ============================================================================
# 启动容器
# ============================================================================

echo ""
echo "🚀 启动容器..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   容器名称: $CONTAINER_NAME"
echo "   GPU: $GPU_ID ($GPU_NAME)"
echo "   端口: $PORT"
echo "   GPU 管理: 已启用 (超时: ${GPU_IDLE_TIMEOUT}秒)"
echo "   API 支持: $ENABLE_API"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 构建启动命令
START_CMD="python app.py \
  --model-path /app/models \
  --model-source local \
  --enable-auto-transcribe \
  --enable-gpu-management \
  --gpu-idle-timeout $GPU_IDLE_TIMEOUT \
  --server-name 0.0.0.0 \
  --server-port 7860"

# 如果启用 API，添加参数
if [ "$ENABLE_API" = true ]; then
    START_CMD="$START_CMD --enable-api"
fi

docker run -d \
  --name $CONTAINER_NAME \
  --restart=always \
  --gpus "device=$GPU_ID" \
  -p $PORT:7860 \
  -v $PROJECT_DIR:/app \
  -v $PROJECT_DIR/models:/app/models:ro \
  -v $PROJECT_DIR/cache:/app/cache \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  step-audio-editx:latest \
  $START_CMD

# ============================================================================
# 等待服务启动
# ============================================================================

echo ""
echo "⏳ 等待服务启动（预计 3 分钟）..."
echo "   提示：首次请求会加载模型到 GPU（20-30秒）"
echo "   提示：任务完成后会自动卸载到 CPU，释放显存"
echo ""

sleep 10

# 显示日志
echo "📋 容器日志（按 Ctrl+C 停止查看）："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker logs -f $CONTAINER_NAME &
LOG_PID=$!

# 等待服务就绪
for i in {1..36}; do
    sleep 5
    if docker logs $CONTAINER_NAME 2>&1 | grep -q "Running on"; then
        kill $LOG_PID 2>/dev/null || true
        break
    fi
done

# ============================================================================
# 显示访问信息
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 服务启动成功！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 访问地址："
echo "   UI 界面: http://0.0.0.0:$PORT"

if [ "$ENABLE_API" = true ]; then
    echo "   API 文档: http://0.0.0.0:$PORT/docs"
    echo "   健康检查: http://0.0.0.0:$PORT/healthz"
fi

echo ""
echo "🎮 GPU 显存管理："
echo "   • 首次请求: 加载模型到 GPU（20-30秒）"
echo "   • 任务完成: 自动卸载到 CPU（2秒，释放显存）"
echo "   • 再次请求: 从 CPU 快速恢复（2-5秒）"
echo "   • 空闲超时: ${GPU_IDLE_TIMEOUT}秒后自动卸载"
echo "   • 手动控制: UI 中的 '🎮 GPU 显存管理' 面板"
echo ""
echo "📊 监控命令："
echo "   查看日志: docker logs -f $CONTAINER_NAME"
echo "   查看 GPU: nvidia-smi"
echo "   进入容器: docker exec -it $CONTAINER_NAME bash"
echo ""
echo "🛑 停止服务："
echo "   docker stop $CONTAINER_NAME"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
