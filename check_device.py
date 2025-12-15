"""
检查设备可用性脚本
"""
import torch

print("=" * 60)
print("设备检测")
print("=" * 60)

print(f"\nPyTorch 版本: {torch.__version__}")

print(f"\nCUDA (NVIDIA GPU):")
print(f"  可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  设备数量: {torch.cuda.device_count()}")
    print(f"  当前设备: {torch.cuda.current_device()}")
    print(f"  设备名称: {torch.cuda.get_device_name(0)}")

print(f"\nMPS (Apple Silicon GPU):")
if hasattr(torch.backends, 'mps'):
    mps_available = torch.backends.mps.is_available()
    print(f"  可用: {mps_available}")
    if mps_available:
        print(f"  ✓ Mac M 系列芯片 GPU 加速可用！")
else:
    print(f"  不可用 (PyTorch 版本可能过低，需要 >= 1.12)")

print(f"\nCPU:")
print(f"  可用: True")

print("\n" + "=" * 60)
print("推荐设备:")
print("=" * 60)

if torch.cuda.is_available():
    print("✓ 使用 CUDA (NVIDIA GPU)")
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("✓ 使用 MPS (Apple Silicon GPU)")
    device = "mps"
else:
    print("⚠ 使用 CPU (无 GPU 加速)")
    device = "cpu"

print(f"\n当前推荐设备: {device}")

# 测试设备
print("\n" + "=" * 60)
print("设备测试")
print("=" * 60)

try:
    test_device = torch.device(device)
    x = torch.randn(3, 3).to(test_device)
    y = x * 2
    print(f"✓ 设备 {device} 测试成功")
    print(f"  测试张量形状: {x.shape}")
except Exception as e:
    print(f"❌ 设备 {device} 测试失败: {str(e)}")
    print("  将回退到 CPU")
    device = "cpu"

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print(f"运行脚本时将使用: {device}")
if device == "mps":
    print("\n💡 Mac M 系列芯片将使用 GPU 加速，性能比 CPU 快 3-5 倍！")
elif device == "cuda":
    print("\n💡 将使用 NVIDIA GPU 加速")
else:
    print("\n⚠️  将使用 CPU，速度较慢")

