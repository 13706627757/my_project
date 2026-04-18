#!/usr/bin/env python3
"""
Fix ONNX models that contain Shape nodes with unsupported 'end' attribute.
This script replaces such Shape nodes with a Constant tensor [1].
"""

import os
import sys


def fix_shape_end(onnx_path: str, output_path: str = None):
    try:
        import onnx
        from onnx import helper, numpy_helper
        import numpy as np
    except ImportError:
        print('Missing required packages: onnx, numpy')
        print('Please install: pip install onnx numpy')
        sys.exit(1)

    if output_path is None:
        output_path = onnx_path

    model = onnx.load(onnx_path)
    graph = model.graph
    nodes = list(graph.node)
    new_nodes = []
    replaced = 0

    for node in nodes:
        if node.op_type == 'Shape':
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if 'end' in attrs or 'start' in attrs:
                print(f"Replacing Shape node '{node.name or '<unnamed>'}' with Constant [1], attrs={attrs}")
                const_tensor = numpy_helper.from_array(np.array([1], dtype=np.int64), name=node.output[0] + '_const')
                const_node = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=[node.output[0]],
                    name=node.name or node.output[0] + '_Const',
                    value=const_tensor,
                )
                new_nodes.append(const_node)
                replaced += 1
                continue
        new_nodes.append(node)

    if replaced == 0:
        print('No Shape nodes with unsupported attributes were found.')
    else:
        graph.ClearField('node')
        graph.node.extend(new_nodes)
        onnx.checker.check_model(model)
        onnx.save(model, output_path)
        print(f'Saved fixed model to {output_path} (replaced {replaced} Shape nodes).')


if __name__ == '__main__':
    onnx_file = 'ONNX/test1.onnx'
    if not os.path.exists(onnx_file):
        print(f'Missing ONNX file: {onnx_file}')
        sys.exit(1)
    print('=' * 60)
    print('Fix ONNX Shape end attribute compatibility')
    print('=' * 60)
    fix_shape_end(onnx_file)
