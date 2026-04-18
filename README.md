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

3. 需要确认 Nano 上安装了 PyTorch，并且 `weights/test1.pt` 文件存在。
