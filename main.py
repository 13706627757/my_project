import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout
from PyQt6.QtCore import QTimer
from UI import GarbageCollectorUI,GarbageCollectorUI__1
from ONNX.example.onnx_example import ONNXModel
import cv2
from PIL import Image
import onnxruntime as rt
import argparse
import serial
import time
import Jetson.GPIO as GPIO
from UI import VideoThread,GPIOThread
import global_vars
def infer():
    print("Entering infer function")
    signal = send_ser.read().decode('ASCII')
    print('receving signal')
    if signal=='T':
        print(f'receve {signal}')
        window.throwRubbish()
        timer.start()
    # v_ret,v_frame=video.read()
    # if v_ret:
    #     window.update_video(v_frame)
    # else :
    #     print("结束,重新播放")
    #     video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    window.update_full()
    if signal == 't':
        print(f'receve {signal}')
        # 从摄像头捕获图像
        ret, frame = camera.read()
        if ret ==False:
            print("摄像头错误")
        print("成功拍照")
        # 将 OpenCV 图像格式转换为 PIL 图像格式
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print("转换图片格式")
        # 获取置信度最高的标签
        # 进行推理
        print("开始推理")
        outputs = model.predict(pil_image)
        print("推理结束")
        predicted_labels = outputs["predictions"]
        if predicted_labels:
            top_prediction = predicted_labels[0]
            label = top_prediction["label"]
            confidence = top_prediction["confidence"]
        #    更新窗口
            type = 0
            window.update(camera_img=frame, label=label,confidence=confidence)
            print(
                f"Top Prediction - Label: {label}, Confidence: {confidence}")
            if label in ["cans",  "bottles"]:
                send_ser.write('0'.encode())  # test_car farword
                print(f'发送信号 {type}')
                response = send_ser.readall()  # read a string from port
                print(response)
            elif label in ["carrots", "potato", "turnip"]:
                send_ser.write('1'.encode())  # test_car farword
                response = send_ser.readall()  # read a string from port
                print(f'发送信号 {type}')
                print(response)
            elif label in ["battery", "medicine"]:
                send_ser.write('2'.encode())  # test_car farword
                response = send_ser.readall()  # read a string from port
                print(f'发送信号 {type}')
                print(response)
            elif label in ["china", "cobble"]:
                send_ser.write('3'.encode())  # test_car farword
                response = send_ser.readall()  # read a string from port
                print(f'发送信号 {type}')
                print(response)
            else:
                send_ser.write('F'.encode())  # test_car stop

    # #读取信号
    # recyclable_signal =GPIO.input(recyclable_pin)
    # foodScrap_signal  =GPIO.input(foodScrap_pin)
    # hazardous_signal  =GPIO.input(hazardous_pin)
    # others_signal     =GPIO.input(others_pin)
    # print(f"\033[92m第一次:{recyclable_signal}  {foodScrap_signal}  {hazardous_signal} {others_signal}\033[92m")
    # time.sleep(0.5)
    # recyclable_signal1 =GPIO.input(recyclable_pin)
    # foodScrap_signal1  =GPIO.input(foodScrap_pin)
    # hazardous_signal1  =GPIO.input(hazardous_pin)
    # others_signal1     =GPIO.input(others_pin)
    # print(f"\033[93m第二次:{recyclable_signal1}  {foodScrap_signal1}  {hazardous_signal1} {others_signal1}\033[93m")
    # if recyclable_signal==0 and recyclable_signal1==0:
    #     print(f'receve recyclable_signal')
    #     print("可回收垃圾满载")
    #     full_flag = window.update(
    #         label=None, camera_img=None, confidence=None, is_full=True, full_type=0)
    # elif foodScrap_signal==0 and foodScrap_signal1==0:
    #     print("厨余垃圾满载")
    #     print(f'receve foodScrap_signal')
    #     full_flag = window.update(
    #         label=None, camera_img=None, confidence=None, is_full=True, full_type=1)
    # elif hazardous_signal==0 and hazardous_signal1==0 :
    #     print(f'receve hazardous_pin')
    #     print("有害垃圾满载")
    #     full_flag = window.update(
    #         label=None, camera_img=None, confidence=None, is_full=True, full_type=2)
    # elif others_signal==0 and others_signal1==0:
    #     print(f'receve others_signal')
    #     print("其他垃圾满载")
    #     full_flag = window.update(
    #         label=None, camera_img=None, confidence=None, is_full=True, full_type=3)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    timer = QTimer()
    # window = GarbageCollectorUI(1366, 500,timer)
    # window.show()
    model_dir = 'ONNX/example'
    camera = cv2.VideoCapture(0)
    video = cv2.VideoCapture('/logo/垃圾分类宣传片2.mp4')
    model = ONNXModel(dir_path=model_dir)
    model.load()
    # receve_ser = serial.serialposix.Serial('/dev/ttyUSB0', 9600, timeout=1)
    send_ser = serial.serialposix.Serial('/dev/ttyUSB0', 9600, timeout=1)
    # 设置GPIO模式为BOARD模式
    GPIO.setmode(GPIO.BOARD)
    # recyclable_signal =1
    # foodScrap_signal  =1
    # hazardous_signal  =1
    # others_signal     =1
    # recyclable_signal1 =1
    # foodScrap_signal1  =1
    # hazardous_signal1  =1
    # others_signal1     =1
    #引脚设置
    recyclable_pin =31
    GPIO.setup(recyclable_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    foodScrap_pin  =33
    GPIO.setup(foodScrap_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    hazardous_pin  =35
    GPIO.setup(hazardous_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    others_pin     =37
    GPIO.setup(others_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    time_singal=0

    window = GarbageCollectorUI__1(800,600,timer,recyclable_pin, foodScrap_pin, hazardous_pin, others_pin)
    # window.showMaximized()
    timer.timeout.connect(infer)  # 连接到update_frame函数
    timer.start(1)  # 每1毫秒更新一次
    window.show()
    sys.exit(app.exec_())