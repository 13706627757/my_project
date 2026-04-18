# my_project
yolov5

## Jetson Nano 直接运行指南

1. 在 Jetson Nano 上进入仓库目录：

```bash
git pull origin main
cd /path/to/lobe
```

2. 直接运行 `.pt` 推理入口：

```bash
python3 infer_pt.py
```

3. 如果 Jetson Nano 不能安装 `ultralytics`，请先在本地 PC 上生成 TorchScript 模型：

```bash
python convert_pt_to_torchscript.py weights/test1.pt weights/test1_ts.pt
```

4. 将生成的 `weights/test1_ts.pt` 复制到 Nano 上，`infer_pt.py` 会优先使用该 TorchScript 模型。

5. 需要确认 Nano 上安装了 PyTorch，并且 `weights/test1.pt` 或 `weights/test1_ts.pt` 文件存在。
