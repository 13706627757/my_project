#!/usr/bin/env python3
"""
将 YOLOv5 .pt 模型转换为 ONNX 格式
在本地 Windows 上运行，生成 ONNX 模型供 Jetson Nano 使用
"""

import sys
import torch

def convert_pt_to_onnx(pt_path: str, onnx_path: str, img_size: int = 640):
    """
    Convert YOLOv5 .pt to ONNX
    
    Args:
        pt_path: 输入的 .pt 文件路径
        onnx_path: 输出的 .onnx 文件路径
        img_size: 输入图像大小
    """
    try:
        print(f"[1/3] 加载 YOLOv5 模型: {pt_path}")
        model = torch.hub.load("ultralytics/yolov5", "custom", path=pt_path, trust_repo=True)
        model.eval()
        
        print(f"[2/3] 创建虚拟输入 ({img_size}x{img_size})")
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        print(f"[3/3] 转换为 ONNX: {onnx_path}")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes={"images": {0: "batch_size"}},
            opset_version=12,
            verbose=False
        )
        
        print(f"✓ 转换成功！ONNX 模型已保存到: {onnx_path}")
        print(f"  文件大小: {__import__('os').path.getsize(onnx_path) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import os
    
    # 输入输出路径
    pt_file = "weights/test1.pt"
    onnx_file = "ONNX/test1.onnx"
    
    if not os.path.exists(pt_file):
        print(f"✗ 找不到 .pt 文件: {pt_file}")
        sys.exit(1)
    
    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)
    
    print("=" * 60)
    print("YOLOv5 .PT -> ONNX 转换工具")
    print("=" * 60)
    
    convert_pt_to_onnx(pt_file, onnx_file)
