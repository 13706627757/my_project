#!/usr/bin/env python3
"""
Jetson Nano 直接使用 YOLOv5 .pt 权重进行推理的入口。
"""

import cv2
import time
from PIL import Image

from yolov5_model import YOLOv5Model
from serial_thread import Serial_thread
from GPIO import GPIOThread


def main():
    print("=" * 60)
    print("垃圾分类系统 - 直接使用 YOLOv5 .pt 推理")
    print("适用于 Jetson Nano，避免 ONNX 兼容性问题")
    print("=" * 60)

    pt_path = 'weights/test1.pt'
    print(f"[INFO] 加载 YOLOv5 权重: {pt_path}")
    model = YOLOv5Model(weights_path=pt_path, conf_thres=0.4)

    print("[INFO] 初始化摄像头...")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] 初始化串口通信...")
    serial_thread = Serial_thread()
    serial_thread.start()

    print("[INFO] 初始化 GPIO...")
    gpio_thread = GPIOThread()
    gpio_thread.start()

    print("[INFO] 系统启动完成，开始推理...")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("[WARNING] 无法读取摄像头帧")
                time.sleep(0.1)
                continue

            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            results = model.predict(pil_image)

            if results["predictions"]:
                best_pred = max(results["predictions"], key=lambda x: x["confidence"])
                label = best_pred["label"]
                confidence = best_pred["confidence"]

                if "recyclable" in label.lower() or "可回收" in label:
                    garbage_type = "recyclable"
                    command = "RECYCLABLE"
                elif "food" in label.lower() or "有害" in label or "hazardous" in label.lower():
                    garbage_type = "hazardous"
                    command = "HAZARDOUS"
                elif "other" in label.lower() or "其他" in label:
                    garbage_type = "others"
                    command = "OTHERS"
                else:
                    garbage_type = "foodScrap"
                    command = "FOODSCRAP"

                print(f"[DETECT] 帧 {frame_count}: {label} -> {garbage_type} (置信度: {confidence:.2f})")
                serial_thread.send_command(command)
                gpio_thread.control_motor(garbage_type)
            else:
                print(f"[INFO] 帧 {frame_count}: 未检测到垃圾")

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                print(f"[INFO] 已处理帧数: {frame_count}, FPS: {fps:.2f}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，正在关闭...")

    finally:
        camera.release()
        serial_thread.stop()
        gpio_thread.stop()
        print("[INFO] 系统已关闭")


if __name__ == '__main__':
    main()
