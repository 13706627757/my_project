#!/usr/bin/env python3
"""
Convert a YOLOv5 .pt weight file into a TorchScript model for Jetson Nano deployment.
This is meant to run on a PC with PyTorch installed, then copy the output file to the Nano.
"""

import os
import sys


def convert_pt_to_torchscript(pt_path: str, output_path: str = None, img_size: int = 640):
    try:
        import torch
    except ImportError:
        print('Please install PyTorch in this environment to convert the model.')
        sys.exit(1)

    if not os.path.exists(pt_path):
        print(f'ERROR: PT file not found: {pt_path}')
        sys.exit(1)

    if output_path is None:
        output_path = os.path.splitext(pt_path)[0] + '_ts.pt'

    print(f'[1/3] Loading weights from: {pt_path}')
    model = None

    try:
        model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=pt_path,
            force_reload=False,
            trust_repo=True,
            _verbose=False,
        )
        print('[INFO] Loaded YOLOv5 model with torch.hub.load')
    except Exception as e:
        print(f'[WARNING] torch.hub.load failed: {e}')
        print('[INFO] Trying direct torch.load or TorchScript load...')
        try:
            model = torch.jit.load(pt_path, map_location='cpu')
            print('[INFO] Loaded TorchScript model directly from .pt file')
        except Exception:
            try:
                model = torch.load(pt_path, map_location='cpu')
                print('[INFO] Loaded model object with torch.load')
            except Exception as e2:
                print(f'[ERROR] Failed to load model from {pt_path}: {e2}')
                sys.exit(1)

    if model is None:
        print('[ERROR] Unable to load the YOLOv5 model from the provided file.')
        sys.exit(1)

    # If the loaded object is a wrapper or model, try to access the underlying module
    try:
        if hasattr(model, 'eval'):
            model.eval()
    except Exception:
        pass

    dummy = torch.randn(1, 3, img_size, img_size)
    print(f'[2/3] Tracing model with dummy input shape {dummy.shape}...')
    try:
        scripted = torch.jit.trace(model, dummy)
        print('[3/3] Saving TorchScript model to:', output_path)
        scripted.save(output_path)
        print('Done. Copy this file to Jetson Nano and use infer_pt.py.')
    except Exception as e:
        print(f'[ERROR] TorchScript tracing failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    pt_file = 'weights/test1.pt'
    output_file = 'weights/test1_ts.pt'
    if len(sys.argv) > 1:
        pt_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    convert_pt_to_torchscript(pt_file, output_file)
