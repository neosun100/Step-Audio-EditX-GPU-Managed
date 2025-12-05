#!/bin/bash

echo "🧪 GPU 显存管理验证脚本"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查容器状态
echo "1️⃣ 检查容器状态"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker ps | grep step-audio-gpu-test
echo ""

# 检查 GPU 管理是否启用
echo "2️⃣ 检查 GPU 管理是否启用"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker logs step-audio-gpu-test 2>&1 | grep "GPU 管理"
echo ""

# 检查初始显存占用
echo "3️⃣ 检查初始显存占用（应该很低，< 5GB）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | grep "^2,"
INITIAL_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
echo "初始显存: ${INITIAL_MEM} MB"
echo ""

# 检查服务是否可访问
echo "4️⃣ 检查服务是否可访问"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s http://localhost:7860 > /dev/null && echo "✅ UI 可访问: http://localhost:7860" || echo "❌ UI 无法访问"
echo ""

# 显示访问信息
echo "5️⃣ 访问信息"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 UI 界面: http://localhost:7860"
echo ""
echo "📊 验证步骤："
echo "   1. 打开 UI 界面"
echo "   2. 找到 '🎮 GPU 显存管理' 面板"
echo "   3. 点击 '🔄 刷新状态' 查看状态"
echo "   4. 上传音频并执行 CLONE 任务"
echo "   5. 观察显存变化（任务完成后应该快速下降）"
echo ""
echo "📈 监控命令："
echo "   watch -n 1 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep \"^2,\"'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 验证脚本执行完成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
