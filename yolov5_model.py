import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import torch


class YOLOv5Model:
    def __init__(self, weights_path: str = "weights/test1.pt", device: str = None, conf_thres: float = 0.25, iou_thres: float = 0.45):
        self.weights_path = weights_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = None
        self.names = {}
        self.load()

    def load(self) -> None:
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(f"YOLOv5 weights file not found: {self.weights_path}")

        # 在 Jetson Nano 上，ultralytics 通常不可用，直接用 torch.hub.load
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=self.weights_path,
            force_reload=False,
            trust_repo=True,
        )

        self.model.to(self.device)
        self.model.conf = self.conf_thres
        self.model.iou = self.iou_thres
        self.names = getattr(self.model, "names", {}) or {}

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        img = np.asarray(image)
        results = self.model(img)

        predictions: List[Dict[str, Any]] = []

        if hasattr(results, "xyxy"):
            raw = results.xyxy[0]
            boxes = raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)
        elif hasattr(results, "pred"):
            raw = results.pred[0]
            boxes = raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)
        else:
            results = results[0]
            if hasattr(results, "boxes"):
                raw = results.boxes.xyxy
                boxes = raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)
            else:
                boxes = np.array([])

        for row in boxes:
            if len(row) < 6:
                continue
            confidence = float(row[4])
            class_id = int(row[5])
            label = self.names.get(class_id, str(class_id))
            predictions.append({"label": label, "confidence": confidence})

        predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
        return {"predictions": predictions}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test YOLOv5 weight loading.")
    parser.add_argument("weights", help="Path to YOLOv5 .pt weights file.")
    parser.add_argument("image", help="Path to an input image.")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(args.image)

    model = YOLOv5Model(weights_path=args.weights)
    image = Image.open(args.image)
    outputs = model.predict(image)
    print(outputs)
