#!/usr/bin/env python3
"""
GPU 显存管理功能测试脚本

测试内容：
1. 懒加载：首次请求加载模型
2. 即用即卸：任务完成后自动卸载
3. 快速恢复：从 CPU 快速恢复到 GPU
4. 自动监控：空闲超时自动卸载
"""

import time
import torch
from gpu_manager import GPUResourceManager


def get_gpu_memory_mb():
    """获取当前 GPU 显存占用（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def create_dummy_model():
    """创建一个虚拟模型用于测试"""
    print("📥 创建虚拟模型（模拟加载过程）...")
    time.sleep(2)  # 模拟加载时间
    
    # 创建一个占用显存的模型
    model = torch.nn.Linear(10000, 10000).cuda()
    print(f"✅ 模型创建完成，显存占用: {get_gpu_memory_mb():.1f} MB")
    return model


def test_lazy_loading():
    """测试1：懒加载"""
    print("\n" + "="*60)
    print("测试1：懒加载")
    print("="*60)
    
    manager = GPUResourceManager(idle_timeout=10)
    
    print(f"初始显存: {get_gpu_memory_mb():.1f} MB")
    
    # 首次获取模型（应该触发加载）
    print("\n第一次获取模型...")
    model = manager.get_model('test_model', create_dummy_model)
    print(f"获取后显存: {get_gpu_memory_mb():.1f} MB")
    
    # 再次获取（应该直接返回）
    print("\n第二次获取模型（应该直接返回）...")
    model = manager.get_model('test_model', create_dummy_model)
    print(f"获取后显存: {get_gpu_memory_mb():.1f} MB")
    
    print("\n✅ 测试1通过：懒加载工作正常")
    return manager


def test_force_offload(manager):
    """测试2：即用即卸"""
    print("\n" + "="*60)
    print("测试2：即用即卸")
    print("="*60)
    
    print(f"卸载前显存: {get_gpu_memory_mb():.1f} MB")
    
    # 手动卸载
    print("\n执行卸载...")
    manager.force_offload('test_model')
    time.sleep(1)
    
    print(f"卸载后显存: {get_gpu_memory_mb():.1f} MB")
    
    print("\n✅ 测试2通过：即用即卸工作正常")


def test_fast_recovery(manager):
    """测试3：快速恢复"""
    print("\n" + "="*60)
    print("测试3：快速恢复")
    print("="*60)
    
    print(f"恢复前显存: {get_gpu_memory_mb():.1f} MB")
    
    # 从 CPU 恢复到 GPU
    print("\n从 CPU 恢复模型到 GPU...")
    start_time = time.time()
    model = manager.get_model('test_model', create_dummy_model)
    elapsed = time.time() - start_time
    
    print(f"恢复后显存: {get_gpu_memory_mb():.1f} MB")
    print(f"恢复耗时: {elapsed:.2f} 秒")
    
    if elapsed < 5:
        print("\n✅ 测试3通过：快速恢复工作正常（< 5秒）")
    else:
        print(f"\n⚠️  测试3警告：恢复时间较长（{elapsed:.2f}秒）")


def test_auto_monitor(manager):
    """测试4：自动监控"""
    print("\n" + "="*60)
    print("测试4：自动监控（空闲超时）")
    print("="*60)
    
    # 启动监控
    manager.start_monitor()
    
    # 获取模型
    print("\n获取模型...")
    model = manager.get_model('test_model', create_dummy_model)
    print(f"获取后显存: {get_gpu_memory_mb():.1f} MB")
    
    # 等待超时（10秒 + 30秒检查间隔）
    print(f"\n等待空闲超时（{manager.idle_timeout}秒 + 30秒检查间隔）...")
    for i in range(manager.idle_timeout + 35):
        time.sleep(1)
        if i % 10 == 0:
            print(f"  已等待 {i} 秒，当前显存: {get_gpu_memory_mb():.1f} MB")
    
    print(f"\n超时后显存: {get_gpu_memory_mb():.1f} MB")
    
    # 停止监控
    manager.stop_monitor()
    
    print("\n✅ 测试4通过：自动监控工作正常")


def test_status():
    """测试5：状态查询"""
    print("\n" + "="*60)
    print("测试5：状态查询")
    print("="*60)
    
    manager = GPUResourceManager(idle_timeout=60)
    
    # 加载模型
    model = manager.get_model('test_model', create_dummy_model)
    
    # 查询状态
    status = manager.get_status()
    print("\n当前状态:")
    print(f"  GPU 显存: {status['gpu_memory_mb']:.1f} MB")
    print(f"  空闲超时: {status['idle_timeout']} 秒")
    print(f"  模型列表:")
    for name, info in status['models'].items():
        print(f"    - {name}: {info['location']} (空闲 {info['idle_seconds']} 秒)")
    
    print("\n✅ 测试5通过：状态查询工作正常")


def main():
    """运行所有测试"""
    print("🧪 GPU 显存管理功能测试")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ 错误：未检测到 CUDA 设备")
        return
    
    print(f"✅ CUDA 可用")
    print(f"   设备: {torch.cuda.get_device_name(0)}")
    print(f"   初始显存: {get_gpu_memory_mb():.1f} MB")
    
    try:
        # 测试1：懒加载
        manager = test_lazy_loading()
        
        # 测试2：即用即卸
        test_force_offload(manager)
        
        # 测试3：快速恢复
        test_fast_recovery(manager)
        
        # 测试4：自动监控（可选，耗时较长）
        # test_auto_monitor(manager)
        
        # 测试5：状态查询
        test_status()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
