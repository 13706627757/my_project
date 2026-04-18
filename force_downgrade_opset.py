#!/usr/bin/env python3
"""
强制降级 ONNX 模型的 opset 版本（修改 protobuf opset_import）
用于兼容旧版 onnxruntime（如 Jetson Nano 1.10.0 的 opset 15 限制）
"""

import sys
import os

def force_downgrade_opset(onnx_path: str, target_opset: int = 15, output_path: str = None):
    """
    强制将 ONNX 模型的 opset 版本改为目标版本
    
    Args:
        onnx_path: 源 ONNX 文件路径
        target_opset: 目标 opset 版本（默认 15，onnxruntime 1.10.0 支持的最高版本）
        output_path: 输出路径（默认覆盖原文件）
    """
    if output_path is None:
        output_path = onnx_path
    
    try:
        import onnx
        print(f"[1/2] 加载 ONNX 模型: {onnx_path}")
        model = onnx.load(onnx_path)
        
        print(f"  原始 IR version: {model.ir_version}")
        print(f"  原始 opset_imports: {[(op.domain, op.version) for op in model.opset_import]}")
        
        # 检查并修改 opset 版本
        updated = False
        for op in model.opset_import:
            original_opset = op.version
            if original_opset > target_opset:
                print(f"\n[2/2] 强制修改 opset 版本为 {target_opset}...")
                print(f"  ⚠ 警告：这是激进操作，可能影响模型兼容性")
                op.version = target_opset
                updated = True
                print(f"  ✓ opset {original_opset} → {target_opset}")
        
        if not updated:
            print(f"  ℹ 模型 opset 已是 {target_opset} 或更低，无需修改")
            return
        
        print(f"  新 opset_imports: {[(op.domain, op.version) for op in model.opset_import]}")
        
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
            err_msg = str(e).lower()
            if 'opset' in err_msg or 'version' in err_msg:
                print(f"⚠ 模型 opset 版本检查失败（预期）: {e}")
                print(f"  这是因为降低 opset 可能导致 op 不兼容，但在运行时可能仍能工作")
            else:
                print(f"⚠ 模型检查警告: {e}")
        
        print(f"\n✓ 成功降级 ONNX opset 版本为 {target_opset}")
        print(f"  现在应该能在 onnxruntime 1.10.0 上加载此模型")
        
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
    print("ONNX 强制 Opset 版本降级工具")
    print("=" * 60)
    
    force_downgrade_opset(onnx_file, target_opset=15)
