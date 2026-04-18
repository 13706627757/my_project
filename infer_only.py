#!/usr/bin/env python3
"""
无 GUI 版本的垃圾分类推理程序
直接运行推理，不需要显示器
"""

import cv2
import time
from PIL import Image
import global_vars
from yolov5_onnx_model import YOLOv5ONNXModel
from serial_thread import Serial_thread
from GPIO import GPIOThread
from infer_thread import InferThread

def main():
    print("=" * 60)
    print("垃圾分类系统 - 无 GUI 版本")
    print("适用于 Jetson Nano 无显示器环境")
    print("=" * 60)

    # 初始化模型
    onnx_path = 'ONNX/test1.onnx'
    print(f"[INFO] 加载 ONNX 模型: {onnx_path}")
    model = YOLOv5ONNXModel(onnx_path=onnx_path, conf_thres=0.4)

    # 初始化摄像头
    print("[INFO] 初始化摄像头...")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    # 设置摄像头参数
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 初始化串口通信
    print("[INFO] 初始化串口通信...")
    serial_thread = Serial_thread()
    serial_thread.start()

    # 初始化 GPIO
    print("[INFO] 初始化 GPIO...")
    gpio_thread = GPIOThread()
    gpio_thread.start()

    print("[INFO] 系统启动完成，开始推理...")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # 读取摄像头帧
            ret, frame = camera.read()
            if not ret:
                print("[WARNING] 无法读取摄像头帧")
                time.sleep(0.1)
                continue

            frame_count += 1

            # 转换为 PIL 图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # 运行推理
            results = model.predict(pil_image)

            # 处理结果
            if results["predictions"]:
                # 找到最高置信度的预测
                best_pred = max(results["predictions"], key=lambda x: x["confidence"])

                # 映射到垃圾分类
                label = best_pred["label"]
                confidence = best_pred["confidence"]

                # 根据标签映射到垃圾类型
                if "recyclable" in label.lower() or "可回收" in label:
                    garbage_type = "recyclable"
                    command = "RECYCLABLE"
                elif "food" in label.lower() or "有害" in label or "hazardous" in label:
                    garbage_type = "hazardous"
                    command = "HAZARDOUS"
                elif "other" in label.lower() or "其他" in label:
                    garbage_type = "others"
                    command = "OTHERS"
                else:
                    garbage_type = "foodScrap"
                    command = "FOODSCRAP"

                print(f"[DETECT] 帧 {frame_count}: {label} -> {garbage_type} (置信度: {confidence:.2f})")

                # 发送串口命令
                serial_thread.send_command(command)

                # 控制 GPIO
                gpio_thread.control_motor(garbage_type)

            else:
                print(f"[INFO] 帧 {frame_count}: 未检测到垃圾")

            # 显示 FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(".1f")

            # 短暂延迟
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，正在关闭...")

    finally:
        # 清理资源
        camera.release()
        serial_thread.stop()
        gpio_thread.stop()
        print("[INFO] 系统已关闭")

if __name__ == "__main__":
    main()