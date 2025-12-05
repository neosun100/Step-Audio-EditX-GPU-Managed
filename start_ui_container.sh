#!/bin/bash
# Step-Audio-EditX UI 容器启动脚本
# 生成时间：2025-11-22
# 用途：启动 UI 服务容器，提供 Gradio Web 界面

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动 Step-Audio-EditX UI 容器"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ⚠️ 配置区域 - 请根据实际情况修改
PROJECT_DIR="/home/neo/upload/Step-Audio-EditX"  # 项目根目录路径
GPU_ID=2                                          # GPU ID (0, 1, 2, 3...)
UI_PORT=7860                                      # 宿主机端口（容器内固定为 7860）

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
if [ "$(docker ps -aq -f name=step-audio-ui-opt)" ]; then
    echo "⚠️  检测到已存在的容器，正在删除..."
    docker stop step-audio-ui-opt 2>/dev/null || true
    docker rm step-audio-ui-opt 2>/dev/null || true
fi

# 检查GPU可用性
echo "📊 检查 GPU 状态..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | while IFS=',' read gpu name total free; do
    echo "   GPU $gpu: $name - Free: ${free}MB / Total: ${total}MB"
done

echo ""
echo "🎯 配置信息："
echo "   - 容器名称：step-audio-ui-opt"
echo "   - GPU：GPU ${GPU_ID}"
echo "   - 端口映射：${UI_PORT} (宿主机) → 7860 (容器)"
echo "   - 重启策略：always"
echo "   - 优化：TF32 + FunASR缓存 + ONNX优化"
echo "   - 项目目录：${PROJECT_DIR}"
echo ""

# 启动容器
docker run -d \
  --name step-audio-ui-opt \
  --restart=always \
  --gpus '"device='${GPU_ID}'"' \
  -p ${UI_PORT}:7860 \
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
    echo "   docker logs -f step-audio-ui-opt"
    echo ""
    echo "🔍 检查容器状态："
    echo "   docker ps | grep step-audio"
    echo ""
    echo "🌐 访问 Web 界面（3分钟后）："
    echo "   http://localhost:${UI_PORT}"
    echo ""
    echo "💡 提示："
    echo "   - 首次启动需要加载模型，请耐心等待"
    echo "   - 看到 'Running on local URL: http://0.0.0.0:7860' 表示启动成功"
    echo ""
else
    echo "❌ 容器启动失败！"
    echo ""
    echo "🔍 排查步骤："
    echo "   1. 检查 Docker 是否运行: docker ps"
    echo "   2. 检查 GPU 是否可用: nvidia-smi"
    echo "   3. 检查镜像是否存在: docker images | grep step-audio-editx"
    echo "   4. 查看错误日志: docker logs step-audio-ui-opt"
    exit 1
fi
