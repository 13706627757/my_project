import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout
from PyQt5.QtCore import QTimer
from ONNX.example.onnx_example import ONNXModel
import cv2
from PIL import Image
import onnxruntime as rt
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import time
from PyQt5.QtCore import QThread, QWaitCondition, QMutex

import global_vars
class InferThread(QThread):
    # 定义一个信号，用于在推理完成后通知主线程
    infer_done_signal = pyqtSignal(np.ndarray, int)
    startExecutionSignal = pyqtSignal()
    def __init__(self, model, camera, serial_thread_0, multi=False):
        super().__init__()
        print("创建推理线程")
        self.net = model
        self.cam = camera
        self.serial_thread_0 = serial_thread_0
        self.img_cuda = None
        self.img_rgb = None
        # self.serial_thread_1 = serial_thread_1
        self.type = 10  # 10代表无类型
        self.is_inferencing = False  # 防止创建多个线程
        #考虑到刚开始不能调用start函数(因为直接调用start函数会连带调用run函数进行推理)
        #因此添加一下的机制变量来使得调用run函数时可以不推理
        self.startExecutionSignal.connect(self.infer)
        self.serial_thread_0.serial_signal0.connect(self.infer_a)
    def run(self):
            # 开始推理线程(不是开始推理)
        self.exec_()
    def infer_a(self):
        singal=self.serial_thread_0.get_singal()
        print(f"在infer_a 中singer {singal}")
        if singal =='a':
            self.infer()
    
    def input(self):
        self.img_cuda, self.img_rgb = self.cam.read()

    def infer(self):
        #开始推理
        print("进入推理函数")
        # if global_vars.is_infering:
        #     print("上一个推理任务还在进行中，跳过这次推理")
        #     return
        # global_vars.is_infering = True
        # print("没有正在推理，开始新的推理")
        # 等待1.5秒,让物品稳定
        time.sleep(2)
        self.input()
        class_idx, _ = self.net.Classify(self.img_cuda)
        # print(f"pil_image.shape:{pil_image.size}")
        # print("转换完成")
        # print("开始推理")
        
        # print("推理结束")
        CLASS_FINAL = {0: {0, 7},   # Y
               1: {2, 3},   # K
               2: {4, 8},   # C
               3: {5, 6, 9}}   # Q
        if class_idx in CLASS_FINAL[0]:
            self.type = 2
            # self.serial_thread_0.serial_0.write('2'.encode())
            print(f'serial_0发送信号 {self.type}')
            #serial_thread_1检测到东西后，发送j使得传送带停止
            # self.serial_thread_1.serial_0.write('j'.encode())
            # print(f"serial_1 发送 j")
            self.serial_thread_0.singal0=''
        # Kitchen Waste
        elif class_idx in CLASS_FINAL[1]:
            self.type = 0
            # self.serial_thread_0.serial_0.write('0'.encode())
            print(f'serial_0 发送信号{self.type}')
            #serial_thread_1检测到东西后，发送j使得传送带停止
            # self.serial_thread_1.serial_0.write('j'.encode())
            self.serial_thread_0.singal0=''
        # Harmful Waste
        elif class_idx in CLASS_FINAL[2]:
            self.type = 1
            # self.serial_thread_0.serial_0.write('1'.encode())
            print(f'serial_0发送信号 {self.type}')
            #serial_thread_1检测到东西后，发送j使得传送带停止
            # self.serial_thread_1.serial_0.write('j'.encode())
            # print(f"serial_1 发送 j")
            self.serial_thread_0.singal0=''
        # Other Waste
        elif class_idx in CLASS_FINAL[3]:
            self.type = 3
            # self.serial_thread_0.serial_0.write('3'.encode())
            print(f'serial_0发送信号 {self.type}')
            #serial_thread_1检测到东西后，发送j使得传送带停止
            # self.serial_thread_1.serial_0.write('j'.encode())
            # print(f"serial_1 发送 j")
            self.serial_thread_0.singal0=''
        
        if self.type != 4:
            self.infer_done_signal.emit(self.img_rgb, self.type)  # 发出信号
        print("推理完成")
        # global_vars.is_infering = False