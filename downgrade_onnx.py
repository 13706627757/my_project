#!/usr/bin/env python3
"""
将导出的 ONNX 模型从高 IR 版本降级到 IR 8（兼容 Jetson Nano onnxruntime 1.10.0）
"""

import sys
import os

def downgrade_onnx_to_ir8(onnx_path: str, output_path: str = None):
    """
    将 ONNX 模型降级到 IR version 8
    
    Args:
        onnx_path: 源 ONNX 文件路径
        output_path: 输出路径（默认覆盖原文件）
    """
    if output_path is None:
        output_path = onnx_path
    
    try:
        import onnx
        print(f"[1/3] 加载 ONNX 模型: {onnx_path}")
        model = onnx.load(onnx_path)
        
        print(f"  原始 IR version: {model.ir_version}")
        print(f"  原始 opset_imports: {[(op.domain, op.version) for op in model.opset_import]}")
        
        # 步骤 1：尝试用 onnx-simplifier 简化模型（减少 op 复杂度）
        print(f"\n[2/3] 尝试用 onnx-simplifier 简化模型...")
        try:
            import onnxsim
            simplified_model, check_ok = onnxsim.simplify(model)
            if check_ok:
                print(f"  ✓ 模型简化成功")
                model = simplified_model
            else:
                print(f"  ⚠ 模型简化后验证失败，继续使用原模型")
        except ImportError:
            print(f"  ⚠ onnx-simplifier 未安装，跳过简化步骤")
            print(f"    可选: pip install onnx-simplifier")
        except Exception as e:
            print(f"  ⚠ 模型简化失败: {e}，继续使用原模型")
        
        # 步骤 2：使用 ONNX version_converter 尝试降到 IR 8
        print(f"\n[3/3] 尝试将 IR 版本降到 8...")
        try:
            # 尝试从当前 IR 版本逐步降到 8
            current_ir = model.ir_version
            target_ir = 8
            
            if current_ir <= target_ir:
                print(f"  ℹ 模型已经是 IR {current_ir}（≤ 8），无需降级")
            else:
                # 尝试直接转换到 IR 8
                print(f"  尝试从 IR {current_ir} 转换到 IR {target_ir}...")
                try:
                    converted_model = onnx.version_converter.convert_version(model, target_ir)
                    model = converted_model
                    print(f"  ✓ 版本转换成功到 IR {target_ir}")
                except Exception as e:
                    print(f"  ⚠ 直接转换失败: {e}")
                    print(f"  尝试逐步降级...")
                    
                    # 尝试逐步降级（从当前版本慢慢降到 8）
                    for target in range(current_ir - 1, target_ir - 1, -1):
                        try:
                            print(f"    尝试转换到 IR {target}...")
                            converted_model = onnx.version_converter.convert_version(model, target)
                            model = converted_model
                            print(f"    ✓ 成功到 IR {target}")
                        except Exception as e:
                            print(f"    ✗ 转换到 IR {target} 失败: {str(e)[:100]}")
                            break
        except Exception as e:
            print(f"  ✗ 版本转换失败: {e}")
            print(f"\n  提示：如果出现 Resize operator 等 op 不兼容，这是 ONNX 工具的限制")
            print(f"        可以尝试: pip install --upgrade onnx")
        
        # 检查最终结果
        print(f"\n[结果] 最终 IR version: {model.ir_version}")
        print(f"  最终 opset_imports: {[(op.domain, op.version) for op in model.opset_import]}")
        
        # 保存模型
        print(f"\n保存模型到: {output_path}")
        onnx.save(model, output_path)
        print(f"✓ 完成！文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        # 验证降级后的模型
        print(f"\n验证模型...")
        try:
            onnx.checker.check_model(output_path)
            print(f"✓ 模型检查通过")
        except Exception as e:
            print(f"⚠ 模型检查警告: {e}")
        
    except ImportError as e:
        print(f"✗ 缺少必要包: {e}")
        print(f"\n请安装:")
        print(f"  pip install onnx")
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
    print("ONNX IR 版本降级工具 (目标: IR 8)")
    print("=" * 60)
    
    downgrade_onnx_to_ir8(onnx_file)
