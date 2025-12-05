#!/bin/bash
# Step-Audio-EditX 统一启动脚本
# UI + API 共享同一个模型实例

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动 Step-Audio-EditX 统一服务器 (UI + API 共享模型)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ⚠️ 配置区域
PROJECT_DIR="/home/neo/upload/Step-Audio-EditX"
GPU_ID=2
PORT=7860

# 检查配置
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ 错误：项目目录不存在: $PROJECT_DIR"
    exit 1
fi

if [ ! -d "$PROJECT_DIR/models" ]; then
    echo "❌ 错误：模型目录不存在: $PROJECT_DIR/models"
    exit 1
fi

# 检查并清理旧容器
CONTAINER_NAME="step-audio-unified"
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "⚠️  检测到已存在的容器，正在删除..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# 检查GPU
echo "📊 检查 GPU 状态..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits | while IFS=',' read gpu name free; do
    echo "   GPU $gpu: $name - Free: ${free}MB"
done

echo ""
echo "🎯 配置信息："
echo "   - 容器名称：$CONTAINER_NAME"
echo "   - GPU：GPU ${GPU_ID}"
echo "   - 端口：${PORT}"
echo "   - 模式：UI + API 统一部署"
echo "   - 共享模型：是"
echo ""

# 启动容器
docker run -d \
  --name $CONTAINER_NAME \
  --restart=always \
  --gpus '"device='${GPU_ID}'"' \
  -p ${PORT}:7860 \
  -v ${PROJECT_DIR}:/app \
  -v ${PROJECT_DIR}/models:/app/models:ro \
  -v ${PROJECT_DIR}/cache:/app/cache \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  step-audio-editx:latest \
  python app.py \
    --model-path /app/models \
    --model-source local \
    --enable-auto-transcribe \
    --enable-api \
    --server-name 0.0.0.0 \
    --server-port 7860

if [ $? -eq 0 ]; then
    echo "✅ 容器启动成功！"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⏳ 等待服务启动（预计 3 分钟）..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📋 查看实时日志："
    echo "   docker logs -f $CONTAINER_NAME"
    echo ""
    echo "🌐 访问服务（3分钟后）："
    echo "   UI 界面: http://localhost:${PORT}"
    echo "   API 文档: http://localhost:${PORT}/docs"
    echo "   健康检查: http://localhost:${PORT}/healthz"
    echo ""
    echo "💡 特性："
    echo "   - UI 和 API 共享同一个模型实例"
    echo "   - 节省 50% 显存占用"
    echo "   - 所有优化功能已启用"
    echo ""
else
    echo "❌ 容器启动失败！"
    exit 1
fi
