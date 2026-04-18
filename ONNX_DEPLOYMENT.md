# YOLOv5 ONNX 部署指南 - Jetson Nano

## 问题
YOLOv5 的 `.pt` 权重文件依赖 `ultralytics` 包加载，但在 Jetson Nano 的 Python 3.6 上无法安装 `ultralytics`。

## 解决方案
将 `.pt` 模型转换为 **ONNX 格式**，使用轻量级 `onnxruntime` 推理，不需要 PyTorch 或 ultralytics。

---

## 步骤 1: 本地转换 (Windows/Mac/Linux)

在本地电脑上安装依赖：
```bash
pip install torch torchvision pillow numpy opencv-python
```

运行转换脚本：
```bash
python convert_pt_to_onnx.py
```

这会生成 `ONNX/test1.onnx` 文件（约 100-200 MB）。

---

## 步骤 2: 上传到 Jetson Nano

将 `ONNX/test1.onnx` 复制到 Jetson Nano 上的相同路径。

**方法 A: 使用 Git**
```bash
git add ONNX/test1.onnx
git commit -m "Add YOLOv5 ONNX model"
git push origin main

# 在 Jetson Nano 上：
cd ~/rubbish_project/my_project
git pull origin main
```

**方法 B: 使用 SCP**
```bash
scp ONNX/test1.onnx jetson@<jetson-ip>:~/rubbish_project/my_project/ONNX/
```

---

## 步骤 3: Jetson Nano 上安装依赖

```bash
# 基础依赖
pip3 install numpy pillow opencv-python pyserial

# ONNX 运行时（很重要）
pip3 install onnxruntime

# PyQt5（如果使用 GUI）
sudo apt-get install python3-pyqt5
```

---

## 步骤 4: 运行应用

```bash
cd ~/rubbish_project/my_project
python3 multi.py
```

---

## 优势

| 方案 | PyTorch | ultralytics | onnxruntime | 包大小 | 速度 |
|-----|---------|-----------|-------------|--------|------|
| YOLOv5 .pt + torch.hub | ✅ | ✅ | ❌ | ~300 MB | 快 |
| **ONNX** | ❌ | ❌ | ✅ | ~50 MB | 中等 |
| ONNX + TensorRT | ❌ | ❌ | ❌ | ~50 MB | 非常快 |

**ONNX 方案在 Jetson Nano 上最可靠：**
- ✅ 不需要 PyTorch（节省 500+ MB）
- ✅ 不需要 ultralytics（避免 Python 3.6 兼容性问题）
- ✅ onnxruntime 轻量级，易于安装
- ✅ 推理速度足够快

---

## 文件说明

- `convert_pt_to_onnx.py` - 本地转换脚本
- `yolov5_onnx_model.py` - ONNX 推理包装器
- `multi.py` - 已改为使用 ONNX 模型
- `ONNX/test1.onnx` - 转换后的模型（需要生成）

---

## 故障排查

**问题 1: onnxruntime 安装失败**
```bash
# 尝试用 conda
conda install -c conda-forge onnxruntime
```

**问题 2: 模型精度下降**
- ONNX 转换时检查 opset_version 和数据类型
- 可能需要调整置信度阈值

**问题 3: 性能不满足**
- 可以改用 TensorRT 进一步优化
- 或减小输入图像分辨率

