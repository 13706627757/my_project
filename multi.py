import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout, QDesktopWidget
from PyQt5.QtCore import QTimer
from video import VideoThread
from UI import GarbageCollectorUI
from yolov5_model import YOLOv5Model
import cv2
from PIL import Image
import argparse
import serial
import time
# import Jetson.GPIO as GPIO
import global_vars
from serial_thread import Serial_thread
from GPIO import GPIOThread
from infer_thread import InferThread
import resnet_model
import resnet
# import lmy_infer
# from utils.jetson_capture import JetsonCapture
# import jetson.inference
# from m_infer import mInferThread#todo
if __name__ == "__main__":
    # 创建pyqt项目
    app = QApplication(sys.argv)
    # 计时器，定时发送信号，运行某个槽函数
    infer_timer = QTimer()
    # 模型路径
    weights_path = 'weights/test1.pt'
    # 摄像头
    camera = cv2.VideoCapture(0)
    # 设置拍摄的大小
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # YOLOv5 模型
    model = YOLOv5Model(weights_path=weights_path, conf_thres=0.4)
    # resnet
    # model = resnet_model.ONNXInferencer("resnet/resnet34.onnx")
    # model=resnet.resnet()
    print("载入模型成功")
    screen = QDesktopWidget().screenGeometry()
    width, height = screen.width(), screen.height()
    # 视频线程
    video_thread = VideoThread(
        "logo/垃圾分类宣传片.mp4")
    # serial串口通信线程
    serial_thread_0 = Serial_thread(global_vars.port0, True)
    serial_thread_0.start()
    # serial_thread_1 = Serial_thread(global_vars.port1,True)
    # 推理线程 resnet/lobe
    infer = InferThread(model, camera, serial_thread_0, True)
    # 主线程
    window = GarbageCollectorUI(
        width-100, height-300, serial_thread_0, infer, video_thread, True)
    video_thread.start()
    window.move(0, 0)
    window.showMaximized()
    # window.show()
    # gpio_thread.start()
    sys.exit(app.exec_())
