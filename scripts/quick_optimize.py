#!/usr/bin/env python3
"""
FunASR 快速优化脚本
立即启用 TF32 和其他简单优化
"""

import os
import sys
import time
import torch
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer import StepAudioTokenizer
from model_loader import ModelSource


def enable_tf32():
    """启用 TF32 加速（Ampere+ GPU）"""
    print("🚀 启用 TF32 加速...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("   ✅ TF32 已启用")


def optimize_onnx_session():
    """返回优化的 ONNX SessionOptions"""
    import onnxruntime
    
    print("🚀 配置优化的 ONNX Runtime...")
    session_option = onnxruntime.SessionOptions()
    
    # 启用所有图优化
    session_option.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    
    # 使用更多线程（之前是 1，太保守）
    session_option.intra_op_num_threads = 4
    session_option.inter_op_num_threads = 2
    
    # 启用并行执行
    session_option.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    
    print("   ✅ ONNX Runtime 优化配置完成")
    print(f"   - 图优化: ENABLE_ALL")
    print(f"   - intra_op 线程: 4")
    print(f"   - inter_op 线程: 2")
    print(f"   - 执行模式: PARALLEL")
    
    return session_option


def benchmark_encoder(tokenizer, test_audio_path, num_runs=3):
    """基准测试编码器"""
    import torchaudio
    
    print(f"\n📊 基准测试编码器（{num_runs} 次运行）...")
    
    # 加载测试音频
    audio, sr = torchaudio.load(test_audio_path)
    audio = audio.cuda()
    
    # 预热
    print("   预热中...")
    _ = tokenizer(audio, sr)
    
    # 测试
    times = []
    for i in range(num_runs):
        start = time.time()
        _ = tokenizer(audio, sr)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   运行 {i+1}/{num_runs}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\n   ✅ 平均耗时: {avg_time:.2f}s")
    
    return avg_time


def test_lightweight_model(encoder_path, test_audio):
    """测试轻量级模型"""
    print("\n🧪 测试轻量级模型: paraformer-base...")
    
    try:
        # 尝试加载 base 模型
        base_model_id = "damo/speech_paraformer-base_asr_nat-zh-cn-16k-common-vocab8404"
        
        tokenizer = StepAudioTokenizer(
            encoder_path=encoder_path,
            model_source=ModelSource.AUTO,
            funasr_model_id=base_model_id
        )
        
        avg_time = benchmark_encoder(tokenizer, test_audio)
        print(f"   ✅ paraformer-base 平均耗时: {avg_time:.2f}s")
        
        return tokenizer, avg_time
        
    except Exception as e:
        print(f"   ⚠️ 无法加载 base 模型: {e}")
        print(f"   提示: 先下载模型到本地")
        return None, None


def apply_torch_compile(model):
    """应用 Torch Compile 优化"""
    print("\n🚀 应用 Torch Compile 优化...")
    
    try:
        import torch
        if hasattr(torch, 'compile'):
            compiled_model = torch.compile(
                model,
                mode="reduce-overhead",
                fullgraph=False  # 兼容性更好
            )
            print("   ✅ Torch Compile 已启用")
            return compiled_model
        else:
            print("   ⚠️ PyTorch 版本不支持 compile (需要 2.0+)")
            return model
    except Exception as e:
        print(f"   ⚠️ Torch Compile 失败: {e}")
        return model


def main():
    parser = argparse.ArgumentParser(description="FunASR 快速优化测试")
    parser.add_argument(
        "--encoder-path",
        default="/model/Step-Audio-Tokenizer",
        help="音频编码器路径"
    )
    parser.add_argument(
        "--test-audio",
        default="/app/examples/zero_shot_en_prompt.wav",
        help="测试音频文件"
    )
    parser.add_argument(
        "--test-lightweight",
        action="store_true",
        help="测试轻量级模型（paraformer-base）"
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=3,
        help="基准测试运行次数"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🎯 FunASR 快速优化测试")
    print("="*80)
    
    # 1. 启用 TF32
    enable_tf32()
    
    # 2. 加载当前模型并测试
    print(f"\n📦 加载当前模型...")
    tokenizer_original = StepAudioTokenizer(
        encoder_path=args.encoder_path,
        model_source=ModelSource.AUTO
    )
    
    baseline_time = benchmark_encoder(
        tokenizer_original,
        args.test_audio,
        args.benchmark_runs
    )
    
    print(f"\n{'='*80}")
    print(f"📊 基线性能（paraformer-large + TF32）")
    print(f"{'='*80}")
    print(f"平均耗时: {baseline_time:.2f}s")
    
    # 3. 测试轻量级模型（可选）
    if args.test_lightweight:
        tokenizer_light, light_time = test_lightweight_model(
            args.encoder_path,
            args.test_audio
        )
        
        if light_time is not None:
            speedup = baseline_time / light_time
            print(f"\n{'='*80}")
            print(f"📊 轻量模型性能（paraformer-base + TF32）")
            print(f"{'='*80}")
            print(f"平均耗时: {light_time:.2f}s")
            print(f"提速: {speedup:.2f}x")
            
            if speedup > 1.5:
                print(f"\n✅ 建议切换到 paraformer-base！")
                print(f"   预期总流程提速: {(baseline_time - light_time) / 24 * 100:.0f}%")
            else:
                print(f"\n⚠️ 提速不明显，保持使用 paraformer-large")
    
    # 4. 总结建议
    print(f"\n{'='*80}")
    print(f"💡 优化建议")
    print(f"{'='*80}")
    
    print("\n立即可做（修改 tokenizer.py）：")
    print("1. 在文件开头添加 TF32 配置：")
    print("   ```python")
    print("   torch.backends.cuda.matmul.allow_tf32 = True")
    print("   torch.backends.cudnn.allow_tf32 = True")
    print("   ```")
    
    print("\n2. 优化 ONNX Runtime 配置（第 62 行）：")
    print("   ```python")
    print("   session_option.intra_op_num_threads = 4  # 从 1 改为 4")
    print("   session_option.inter_op_num_threads = 2")
    print("   session_option.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL")
    print("   ```")
    
    if args.test_lightweight:
        print("\n3. 切换到轻量模型（第 22 行）：")
        print("   ```python")
        print('   funasr_model_id="damo/speech_paraformer-base_asr_nat-zh-cn-16k-common-vocab8404"')
        print("   ```")
    
    print("\n预期总体效果：")
    estimated_speedup = 1.2  # TF32 + ONNX 优化
    if args.test_lightweight and light_time is not None:
        estimated_speedup = baseline_time / light_time
    
    new_total_time = 24 - (baseline_time - baseline_time / estimated_speedup)
    improvement = (24 - new_total_time) / 24 * 100
    
    print(f"  - 编码时间: {baseline_time:.1f}s → {baseline_time / estimated_speedup:.1f}s")
    print(f"  - 总流程时间: 24s → {new_total_time:.1f}s")
    print(f"  - 性能提升: {improvement:.0f}%")
    
    print(f"\n📚 详细优化指南:")
    print(f"  - docs/funasr-optimization-guide.md")


if __name__ == "__main__":
    main()
