import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout
from PyQt5.QtCore import QTimer
from UI import GarbageCollectorUI
from ONNX.example.onnx_example import ONNXModel
import cv2
from PIL import Image
import onnxruntime as rt
import argparse
import serial
import time
import Jetson.GPIO as GPIO
from UI import VideoThread, GPIOThread
import global_vars
from serial_thread import Serial_thread
from infer import InferThread
if __name__ == "__main__":
    app = QApplication(sys.argv)
    infer_timer = QTimer()
    model_dir = 'ONNX/example'
    camera = cv2.VideoCapture(1)
    video = cv2.VideoCapture('/logo/垃圾分类宣传片2.mp4')
    model = ONNXModel(dir_path=model_dir)
    model.load()
    print("载入模型成功")
    # 设置GPIO模式为BOARD模式
    GPIO.setmode(GPIO.BOARD)
    recyclable_pin = 31
    foodScrap_pin = 33
    hazardous_pin = 35
    others_pin = 37
    # serial串口通信线程
    serial_thread = Serial_thread()
    # 推理线程
    infer = InferThread(model, camera, serial_thread.serial_0)
    # GPIO线程
    gpio_thread = GPIOThread(
        recyclable_pin, foodScrap_pin, hazardous_pin, others_pin)
    window = GarbageCollectorUI(800, 600, serial_thread, infer, gpio_thread)
    # window.show()
    window.showMaximized()
    serial_thread.start()
    gpio_thread.start()
    sys.exit(app.exec_())
