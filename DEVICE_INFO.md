# 设备加速说明

## Mac M 系列芯片 GPU 加速

### 支持情况

Mac M 系列芯片（M1/M2/M3/M4/M5）支持使用 **MPS (Metal Performance Shaders)** 进行 GPU 加速。

### 自动检测

代码已自动配置为：
1. **优先使用 CUDA**（如果有 NVIDIA GPU）
2. **其次使用 MPS**（如果是 Mac M 系列芯片）
3. **最后使用 CPU**

### 验证是否使用 MPS

运行测试或训练时，会显示：
```
使用设备: mps
```

如果显示 `cpu`，可能是：
- PyTorch 版本过低（需要 >= 1.12）
- MPS 不可用

### 检查 MPS 是否可用

```python
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"MPS 可用: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
```

### 性能提升

使用 MPS 相比 CPU 通常可以：
- **推理速度**: 提升 3-5 倍
- **训练速度**: 提升 2-4 倍
- **内存效率**: 更好的内存管理

### 注意事项

1. **fp16 支持**: MPS 暂不支持混合精度训练（fp16），代码会自动禁用
2. **某些操作**: 部分 PyTorch 操作可能不支持 MPS，会自动回退到 CPU
3. **内存**: MPS 使用统一内存架构，通常比 CUDA 更高效

### 手动指定设备

如果需要强制使用特定设备：

```python
# 强制使用 MPS
import torch
device = torch.device("mps")

# 强制使用 CPU
device = torch.device("cpu")
```

### 常见问题

**Q: 为什么显示使用 CPU 而不是 MPS？**

A: 可能原因：
1. PyTorch 版本 < 1.12（需要升级）
2. macOS 版本过低
3. 某些操作不支持 MPS，自动回退

**Q: MPS 和 CUDA 性能对比？**

A: 
- CUDA (NVIDIA GPU): 通常最快，支持最完整
- MPS (Apple Silicon): 在 Mac 上性能优秀，但某些操作可能不支持
- CPU: 最慢但最稳定

**Q: 如何升级 PyTorch 以支持 MPS？**

A:
```bash
pip install --upgrade torch torchvision torchaudio
```

### 性能测试

运行测试脚本时，MPS 会显著加速：

```bash
# 使用 MPS 加速（自动）
python test_pretrained.py --model_key 1

# 训练时也会自动使用 MPS
python train_with_hp_tuning.py --model_key 1
```

### 总结

✅ **Mac M 系列芯片会自动使用 MPS GPU 加速**
✅ **无需额外配置，代码已自动检测**
✅ **性能比 CPU 快 3-5 倍**

运行任何脚本时，如果检测到 Mac M 系列芯片，会自动使用 MPS 加速！

