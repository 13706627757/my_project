"""
ONNX 模型推理包装器 - 用于 YOLOv5 ONNX 模型
在 Jetson Nano 上运行，只需要 onnxruntime，不依赖 PyTorch
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import cv2

try:
    import onnxruntime as rt
except ImportError:
    raise ImportError("Please install onnxruntime: pip install onnxruntime")


class YOLOv5ONNXModel:
    """ONNX 版本的 YOLOv5 模型加载和推理"""
    
    def __init__(self, onnx_path: str = "ONNX/test1.onnx", conf_thres: float = 0.25, iou_thres: float = 0.45):
        self.onnx_path = onnx_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.session = None
        self.input_name = None
        self.output_name = None
        self.img_size = 640
        self.names = {}
        
        self.load()
    
    def load(self) -> None:
        """加载 ONNX 模型"""
        if not os.path.isfile(self.onnx_path):
            raise FileNotFoundError(f"ONNX 模型文件不存在: {self.onnx_path}")
        
        print(f"[INFO] 加载 ONNX 模型: {self.onnx_path}")
        
        # 使用 CPU 执行（Jetson Nano 上 GPU 支持需要特殊配置）
        providers = ['CPUExecutionProvider']
        # 如果有 CUDA，尝试使用
        try:
            if 'CUDAExecutionProvider' in rt.get_available_providers():
                providers = ['CUDAExecutionProvider'] + providers
                print("[INFO] 使用 CUDA 加速")
        except:
            pass
        
        try:
            self.session = rt.InferenceSession(self.onnx_path, providers=providers)
        except Exception as e:
            print(f"[WARNING] onnxruntime 加载失败: {e}")
            print(f"[WARNING] 当前 onnxruntime 版本: {getattr(rt, '__version__', 'unknown')}")
            print("[WARNING] 如果出现 Unsupported model IR version，请使用 opset_version=10 或 9 重新导出模型，")
            print("          并确认当前 Python 环境加载的是你实际使用的 onnxruntime。")
            raise
        
        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"[SUCCESS] ONNX 模型加载成功")
    
    def preprocess(self, image: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
        """预处理图像"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        orig_size = image.size
        
        # 调整大小到 640x640
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # 转为 numpy 并归一化
        img_array = np.asarray(image, dtype=np.float32) / 255.0
        
        # CHW 格式
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # 添加 batch 维度
        img_array = np.expand_dims(img_array, 0)
        
        return img_array, orig_size
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        运行推理
        
        Returns:
            {
                "predictions": [
                    {"label": "class_name", "confidence": 0.95, "box": [x1, y1, x2, y2]},
                    ...
                ]
            }
        """
        img_array, orig_size = self.preprocess(image)
        
        # ONNX 推理
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: img_array}
        )
        
        # 解析输出
        predictions = self._parse_output(outputs[0], orig_size)
        
        return {"predictions": predictions}
    
    def _parse_output(self, output: np.ndarray, orig_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        解析 ONNX 输出
        
        YOLOv5 ONNX 输出格式: (1, 25200, 85) 或 (1, 25200, n_classes+5)
        每个检测: [x, y, w, h, conf, class_0, class_1, ...]
        """
        predictions = []
        
        # output shape: (1, 25200, 85) 或类似
        output = output[0]  # 去掉 batch 维度
        
        # 过滤置信度
        conf_mask = output[:, 4] > self.conf_thres
        detections = output[conf_mask]
        
        if len(detections) == 0:
            return predictions
        
        # 获取最高置信度的类
        class_scores = detections[:, 5:]
        max_class_idx = np.argmax(class_scores, axis=1)
        max_class_conf = np.max(class_scores, axis=1)
        
        # 过滤双重置信度
        double_conf_mask = max_class_conf > self.conf_thres
        detections = detections[double_conf_mask]
        max_class_idx = max_class_idx[double_conf_mask]
        max_class_conf = max_class_conf[double_conf_mask]
        
        # NMS（简化版）
        if len(detections) > 0:
            # YOLOv5 默认类名（如果没有专门标签）
            class_names = self.names or {i: f"class_{i}" for i in range(80)}
            
            # 缩放到原始尺寸
            h_scale = orig_size[1] / self.img_size
            w_scale = orig_size[0] / self.img_size
            
            for i, det in enumerate(detections):
                x_center, y_center, w, h = det[:4]
                conf = det[4]
                class_id = max_class_idx[i]
                class_conf = max_class_conf[i]
                
                # 转换为像素坐标
                x1 = (x_center - w / 2) * w_scale
                y1 = (y_center - h / 2) * h_scale
                x2 = (x_center + w / 2) * w_scale
                y2 = (y_center + h / 2) * h_scale
                
                predictions.append({
                    "label": class_names.get(class_id, f"class_{class_id}"),
                    "confidence": float(class_conf),
                    "box": [float(x1), float(y1), float(x2), float(y2)]
                })
        
        return predictions
