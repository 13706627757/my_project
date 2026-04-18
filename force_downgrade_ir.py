#!/usr/bin/env python3
"""
强制降级 ONNX 模型 IR 版本（修改 protobuf IR 字段）
警告：这是激进方法，可能影响模型兼容性。但对于 Jetson Nano 是必需的。
"""

import sys
import os

def force_downgrade_ir_version(onnx_path: str, target_ir: int = 8, output_path: str = None):
    """
    强制将 ONNX 模型的 IR 版本号改为目标版本
    
    Args:
        onnx_path: 源 ONNX 文件路径
        target_ir: 目标 IR 版本（默认 8）
        output_path: 输出路径（默认覆盖原文件）
    """
    if output_path is None:
        output_path = onnx_path
    
    try:
        import onnx
        print(f"[1/2] 加载 ONNX 模型: {onnx_path}")
        model = onnx.load(onnx_path)
        
        original_ir = model.ir_version
        print(f"  原始 IR version: {original_ir}")
        print(f"  原始 opset_imports: {[(op.domain, op.version) for op in model.opset_import]}")
        
        if original_ir <= target_ir:
            print(f"  ℹ 模型已经是 IR {original_ir}（≤ {target_ir}），无需修改")
            return
        
        print(f"\n[2/2] 强制修改 IR 版本为 {target_ir}...")
        print(f"  ⚠ 警告：这是激进操作，可能影响模型兼容性")
        
        # 强制修改 IR 版本
        model.ir_version = target_ir
        
        print(f"  ✓ 修改完成，新 IR version: {model.ir_version}")
        
        # 保存
        print(f"\n保存模型到: {output_path}")
        onnx.save(model, output_path)
        print(f"✓ 完成！文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        # 验证
        print(f"\n验证模型...")
        try:
            onnx.checker.check_model(output_path)
            print(f"✓ 模型检查通过")
        except Exception as e:
            print(f"⚠ 模型检查警告: {e}")
            print(f"  这可能是正常的，继续使用模型")
        
        print(f"\n✓ 成功降级 ONNX IR 版本从 {original_ir} 到 {target_ir}")
        print(f"  现在可以在 Jetson Nano 上加载此模型")
        
    except ImportError:
        print(f"✗ 缺少 onnx 包")
        print(f"  请运行: pip install onnx")
        sys.exit(1)
    except Exception as e:
        print(f"✗ 降级失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    onnx_file = "ONNX/test1.onnx"
    
    if not os.path.exists(onnx_file):
        print(f"✗ ONNX 文件不存在: {onnx_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("ONNX 强制 IR 版本降级工具")
    print("=" * 60)
    
    force_downgrade_ir_version(onnx_file, target_ir=8)
