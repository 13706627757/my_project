#!/usr/bin/env python3
"""
将 YOLOv5 .pt 模型转换为 ONNX 格式
在本地 Windows 上运行，生成 ONNX 模型供 Jetson Nano 使用
"""

import sys
import torch

def convert_pt_to_onnx(pt_path: str, onnx_path: str, img_size: int = 640):
    """
    Convert YOLOv5 .pt to ONNX using official export method
    
    Args:
        pt_path: 输入的 .pt 文件路径
        onnx_path: 输出的 .onnx 文件路径
        img_size: 输入图像大小
    """
    try:
        print(f"[1/2] 加载 YOLOv5 模型: {pt_path}")
        model = torch.hub.load("ultralytics/yolov5", "custom", path=pt_path, trust_repo=True)
        model.eval()
        
        # 方案 1: 使用 YOLOv5 官方的 export 方法（如果有）
        if hasattr(model, 'export'):
            print(f"[2/2] 使用 YOLOv5 官方导出方法转换为 ONNX...")
            try:
                model.export(format='onnx', imgsz=img_size)
                # 官方导出会生成到同目录
                import shutil
                pt_dir = os.path.dirname(pt_path)
                pt_name = os.path.splitext(os.path.basename(pt_path))[0]
                official_onnx = os.path.join(pt_dir, f'{pt_name}.onnx')
                if os.path.exists(official_onnx):
                    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
                    shutil.move(official_onnx, onnx_path)
                    print(f"✓ 转换成功！ONNX 模型已保存到: {onnx_path}")
                    print(f"  文件大小: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
                    return
            except Exception as e:
                print(f"  官方导出失败，尝试备选方案: {e}")
        
        # 方案 2: 使用 torch.onnx.export（需要 onnx 包）
        print(f"[2/2] 使用 PyTorch ONNX 导出...")
        print(f"  创建虚拟输入 ({img_size}x{img_size})")
        
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes={"images": {0: "batch_size"}},
            opset_version=11,
            verbose=False,
            do_constant_folding=True
        )
        
        print(f"✓ 转换成功！ONNX 模型已保存到: {onnx_path}")
        print(f"  文件大小: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        print(f"\n提示：如果出现 onnxscript 错误，请运行:")
        print(f"  conda activate yolov5-webcam")
        print(f"  conda install onnx -y")
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
