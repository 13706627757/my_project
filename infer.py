import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout
from PyQt5.QtCore import QTimer
from ONNX.example.onnx_example import ONNXModel
import cv2
from PIL import Image
import onnxruntime as rt
import Jetson.GPIO as GPIO
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np


class InferThread(QThread):
    # 定义一个信号，用于在推理完成后通知主线程
    infer_done_signal = pyqtSignal(np.ndarray, int)

    def __init__(self, model, camera, serial_0):
        super().__init__()
        self.model = model
        self.camera = camera
        self.serial_0 = serial_0
        self.type = 10  # 10代表无类型
        self.is_inferencing = False  # 防止创建多个线程

    def run(self):
        # self.is_inferencing = True
        print("Entering infer function")
        # 多拍几张  刷新缓冲区
        _, frame = self.camera.read()
        _, frame = self.camera.read()
        _, frame = self.camera.read()
        _, frame = self.camera.read()
        ret, frame = self.camera.read()
        # print(frame.shape)
        frame = frame[700:-170, 300:, :]
        frame = frame.astype(np.uint8)
        # print(frame.shape)
        frame
        if not ret:
            print("摄像头错误")
            return
        print("成功拍照")
        print("转换图片格式")
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print("转换完成")
        print("开始推理")
        outputs = self.model.predict(pil_image)
        print("推理结束")
        top_prediction = outputs["predictions"][0]
        label = top_prediction["label"]
        confidence = top_prediction["confidence"]
        print(f"Top Prediction - Label: {label}, Confidence: {confidence}")
        if label in ["cans",  "bottles"]:
            self.type = 0
            self.serial_0.write('0'.encode())
            print(f'发送信号 {self.type}')
        elif label in ["carrots", "potato", "turnip"]:
            self.type = 1
            self.serial_0.write('1'.encode())
            print(f'发送信号 {self.type}')
        elif label in ["battery", "medicine"]:
            self.type = 2
            self.serial_0.write('2'.encode())
            print(f'发送信号 {self.type}')
        elif label in ["china", "cobble"]:
            self.type = 3
            self.serial_0.write('3'.encode())
            print(f'发送信号 {self.type}')
        self.infer_done_signal.emit(frame, self.type)  # 发出信号
        self.is_inferencing = False
