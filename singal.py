import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout, QDesktopWidget
from PyQt5.QtCore import QTimer
from UI import GarbageCollectorUI
from ONNX.example.onnx_example import ONNXModel
import cv2
from PIL import Image
import onnxruntime as rt
import argparse
import serial
import time
import global_vars
from serial_thread import Serial_thread
from infer_thread import InferThread
from video import VideoThread
if __name__ == "__main__":
    app = QApplication(sys.argv)
    infer_timer = QTimer()
    model_dir = 'ONNX/example'
    camera = cv2.VideoCapture(0)
    video = cv2.VideoCapture('/logo/垃圾分类宣传片2.mp4')
    model = ONNXModel(dir_path=model_dir)
    model.load()
    print("载入模型成功")
    # serial串口通信线程
    serial_thread_0 = Serial_thread(global_vars.port0,True)
    # serial_thread_1 = Serial_thread(global_vars.port1,True)
    # 推理线程
    infer = InferThread(model, camera, serial_thread_0)
    # 获取屏幕大小
    screen = QDesktopWidget().screenGeometry()
    width, height = screen.width(), screen.height()
    # 视频线程
    video_thread = VideoThread(
        "logo/垃圾分类宣传片.mp4")
    video_thread.start()
    window = GarbageCollectorUI(
        width-100, height-100, serial_thread_0,  infer, video_thread)
    # window.show()
    window.showMaximized()
    serial_thread_0.start()
    sys.exit(app.exec_())
