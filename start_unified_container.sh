#!/bin/bash
# Step-Audio-EditX 统一容器启动脚本
# 功能：UI + API 统一部署，共享模型实例，支持懒加载
# 生成时间：2025-12-03

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动 Step-Audio-EditX 统一容器 (UI + API + 懒加载)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ⚠️ 配置区域 - 请根据实际情况修改
PROJECT_DIR="/home/neo/upload/Step-Audio-EditX"  # 项目根目录路径
GPU_ID=2                                          # GPU ID (0, 1, 2, 3...)
PORT=7860                                         # 服务端口（UI和API共用）
IDLE_TIMEOUT=300                                  # 模型空闲超时（秒），默认300秒（5分钟）

# 检查配置
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ 错误：项目目录不存在: $PROJECT_DIR"
    echo "   请修改脚本中的 PROJECT_DIR 变量"
    exit 1
fi

if [ ! -d "$PROJECT_DIR/models" ]; then
    echo "❌ 错误：模型目录不存在: $PROJECT_DIR/models"
    echo "   请确保已下载模型到 models/ 目录"
    exit 1
fi

# 检查是否已存在容器
CONTAINER_NAME="step-audio-unified"
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "⚠️  检测到已存在的容器，正在删除..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# 检查GPU可用性
echo "📊 检查 GPU 状态..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | while IFS=',' read gpu name total free; do
    echo "   GPU $gpu: $name - Free: ${free}MB / Total: ${total}MB"
done

echo ""
echo "🎯 配置信息："
echo "   - 容器名称：$CONTAINER_NAME"
echo "   - GPU：GPU ${GPU_ID}"
echo "   - 端口：${PORT} (UI + API)"
echo "   - 重启策略：always"
echo "   - 懒加载：启用"
echo "   - 空闲超时：${IDLE_TIMEOUT}秒"
echo "   - 优化：TF32 + FunASR缓存 + ONNX优化"
echo "   - 项目目录：${PROJECT_DIR}"
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
  -e OMP_NUM_THREADS=8 \
  -e MKL_NUM_THREADS=8 \
  step-audio-editx:latest \
  python server.py \
    --model-path /app/models \
    --model-source local \
    --enable-auto-transcribe \
    --idle-timeout ${IDLE_TIMEOUT} \
    --host 0.0.0.0 \
    --port 7860

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
    echo "🔍 检查容器状态："
    echo "   docker ps | grep $CONTAINER_NAME"
    echo ""
    echo "🌐 访问服务（3分钟后）："
    echo "   UI 界面: http://localhost:${PORT}"
    echo "   API 文档: http://localhost:${PORT}/docs"
    echo "   健康检查: http://localhost:${PORT}/healthz"
    echo "   模型状态: http://localhost:${PORT}/api/v1/models/status"
    echo ""
    echo "💡 懒加载特性："
    echo "   - 模型按需加载，首次使用时才会加载到GPU"
    echo "   - 空闲 ${IDLE_TIMEOUT} 秒后自动卸载，释放显存"
    echo "   - 支持 base/awq/bnb 三种模型变体"
    echo "   - UI 和 API 共享同一个模型实例"
    echo ""
    echo "🔧 手动卸载模型："
    echo "   curl -X POST http://localhost:${PORT}/api/v1/models/base/unload"
    echo ""
else
    echo "❌ 容器启动失败！"
    echo ""
    echo "🔍 排查步骤："
    echo "   1. 检查 Docker 是否运行: docker ps"
    echo "   2. 检查 GPU 是否可用: nvidia-smi"
    echo "   3. 检查镜像是否存在: docker images | grep step-audio-editx"
    echo "   4. 查看错误日志: docker logs $CONTAINER_NAME"
    exit 1
fi
