#!/usr/bin/env python3
"""
Clean unsupported MaxPool attributes from an ONNX model.
This is meant for older onnxruntime versions like Jetson Nano 1.10.0.
"""

import sys
import os

def clean_maxpool_attributes(onnx_path: str, output_path: str = None):
    try:
        import onnx
        from onnx import helper
    except ImportError:
        print('Missing required package: onnx')
        print('Install with: pip install onnx')
        sys.exit(1)

    if output_path is None:
        output_path = onnx_path

    model = onnx.load(onnx_path)
    removed = 0
    if not model.graph.node:
        print('No nodes found in model.')
        sys.exit(1)

    for node in model.graph.node:
        if node.op_type == 'MaxPool':
            attrs = list(node.attribute)
            keep = []
            for attr in attrs:
                if attr.name in ('ceil_mode', 'storage_order'):
                    print(f"Removing unsupported attribute '{attr.name}' from {node.name}")
                    removed += 1
                    continue
                keep.append(attr)
            node.ClearField('attribute')
            node.attribute.extend(keep)

    if removed == 0:
        print('No unsupported MaxPool attributes found.')
    else:
        onnx.save(model, output_path)
        print(f'Saved cleaned model to {output_path}. Removed {removed} attributes.')
        try:
            onnx.checker.check_model(model)
            print('ONNX model is valid after cleanup.')
        except Exception as e:
            print(f'Warning: model check failed after cleanup: {e}')

if __name__ == '__main__':
    onnx_file = 'ONNX/test1.onnx'
    if not os.path.exists(onnx_file):
        print(f'ONNX file not found: {onnx_file}')
        sys.exit(1)
    clean_maxpool_attributes(onnx_file)
