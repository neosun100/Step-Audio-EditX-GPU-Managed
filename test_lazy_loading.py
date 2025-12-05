#!/usr/bin/env python3
"""
测试懒加载功能
"""
import time
import torch
from lazy_model_manager import LazyModelManager


def dummy_model_factory():
    """模拟模型工厂"""
    print("🔄 正在加载模型...")
    time.sleep(2)  # 模拟加载时间
    
    # 创建一个简单的模型（占用一些显存）
    model = torch.nn.Linear(1000, 1000).cuda()
    print("✅ 模型加载完成")
    return model


def test_lazy_loading():
    """测试懒加载"""
    print("=" * 80)
    print("测试懒加载管理器")
    print("=" * 80)
    
    # 创建管理器
    print("\n1. 创建懒加载管理器（空闲超时: 10秒）")
    manager = LazyModelManager(
        model_factory=dummy_model_factory,
        idle_timeout=10,
        auto_unload=True
    )
    
    # 检查初始状态
    print("\n2. 检查初始状态")
    status = manager.get_status()
    print(f"   模型已加载: {status['loaded']}")
    if torch.cuda.is_available():
        print(f"   GPU显存: {status.get('gpu_memory_allocated_gb', 0):.2f} GB")
    
    # 首次获取模型（触发加载）
    print("\n3. 首次获取模型（触发加载）")
    model = manager.get_model()
    print(f"   模型类型: {type(model)}")
    
    # 检查加载后状态
    print("\n4. 检查加载后状态")
    status = manager.get_status()
    print(f"   模型已加载: {status['loaded']}")
    if torch.cuda.is_available():
        print(f"   GPU显存: {status.get('gpu_memory_allocated_gb', 0):.2f} GB")
    
    # 再次获取模型（不会重新加载）
    print("\n5. 再次获取模型（不会重新加载）")
    model2 = manager.get_model()
    print(f"   是同一个模型: {model is model2}")
    
    # 等待空闲超时
    print("\n6. 等待空闲超时（10秒）...")
    for i in range(10):
        time.sleep(1)
        status = manager.get_status()
        if status['loaded']:
            idle_time = status.get('idle_time', 0)
            time_until_unload = status.get('time_until_unload', 0)
            print(f"   [{i+1}/10] 空闲时间: {idle_time:.1f}秒, 距离卸载: {time_until_unload:.1f}秒")
        else:
            print(f"   [{i+1}/10] 模型已卸载")
            break
    
    # 等待卸载完成
    time.sleep(2)
    
    # 检查卸载后状态
    print("\n7. 检查卸载后状态")
    status = manager.get_status()
    print(f"   模型已加载: {status['loaded']}")
    if torch.cuda.is_available():
        print(f"   GPU显存: {status.get('gpu_memory_allocated_gb', 0):.2f} GB")
    
    # 再次获取模型（触发重新加载）
    print("\n8. 再次获取模型（触发重新加载）")
    model3 = manager.get_model()
    print(f"   模型类型: {type(model3)}")
    
    # 手动卸载
    print("\n9. 手动卸载模型")
    manager.force_unload()
    status = manager.get_status()
    print(f"   模型已加载: {status['loaded']}")
    
    # 关闭管理器
    print("\n10. 关闭管理器")
    manager.shutdown()
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️  警告：未检测到CUDA，将使用CPU模式测试")
    
    test_lazy_loading()
