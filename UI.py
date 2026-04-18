import time
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer
import cv2
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QUrl
import os
from PyQt5.QtCore import QThread, pyqtSignal
from infer_thread import InferThread
from video import VideoThread
import global_vars


class GarbageCollectorUI(QMainWindow):
    def __init__(self, width, height, serial_thread0, infer_thread, video_thread, multi=False):
        super().__init__()
        # 创建视频播放线程
        self.video_thread = video_thread
        self.video_thread.video_signal.connect(self.update_video_slot)
        # 推理线程
        self.infer_thread = infer_thread
        self.infer_thread.start()
        self.infer_thread.infer_done_signal.connect(self.update)
        # serial通信线程
        self.serial_thread0 = serial_thread0
        self.serial_thread0.serial_signal0.connect(self.serial_signal0)
        # 计时推理
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.start_inference_a)
        self.timer.start(500)  # 单位毫秒

        # 用于更新多垃圾
        self.update_list_index = 0
        self.throw_num = 0
        self.currentNum = 0
        self.update_times = 1
        # self.serial_thread1 = serial_thread1
        # self.serial_thread1.serial_signal0.connect(
        # self.start_inference_t)  # 从1接受't'开始推理
        self.multi = multi  # 垃圾模式:multi为True时是多垃圾
        # self.serial_signal_up_raster: str = ''  # 与单片机0-上光栅控
        self.toUpdate = False   # 用于更新检测信息
        self.new_data: str = ''  # 更新右上角的信息
        # 设置主窗口大小
        self.resize(width, height)
        self.totalNumber = 0
        # 主布局
        main_layout = QGridLayout()  # 使用 QGridLayout 作为主布局
        # 左上角
        self.video_image = None  # 初始化为 None
        self.video_label = QLabel(self)
        main_layout.addWidget(self.video_label, 0, 0)  # 添加到主布局的左上角

        # 右上角：显示最近数据和历史数据
        self.QtotalNum = QLabel(f"垃圾总数:{self.totalNumber}")
        # 设置QtotalNum的字体大小和加粗
        top_right_font = QFont()
        top_right_font.setPointSize(25)  # 设置字体大小
        top_right_font.setBold(True)    # 设置字体加粗
        top_right_layout = QVBoxLayout()
        self.QtotalNum.setFont(top_right_font)
        self.recent_data_label = QTextEdit(self)
        self.recent_data_label.setFont(top_right_font)
        self.recent_data_label.setReadOnly(True)  # 设置为只读，因为我们不希望用户编辑它
        top_right_layout.addWidget(self.QtotalNum)
        top_right_layout.addWidget(self.recent_data_label)
        top_right_widget = QWidget()
        top_right_widget.setLayout(top_right_layout)
        main_layout.addWidget(top_right_widget, 0, 1)  # 添加到主布局的右上角

        # 左下角：显示四张图片和文字
        left_bottom_font = QFont()
        left_bottom_font.setPointSize(20)
        top_right_font.setBold(True)
        self.numbers = [0, 0, 0, 0]
        self.garbage_types = ["可回收垃圾", "厨余垃圾", "有害垃圾", "其他垃圾"]
        self.type = 10
        self.image_names = ["recyclable.png", "foodScrap.png",
                            "hazardous.png", "others.png"]
        self.garbage_text_labels = []  # 左下角的文字
        self.full = 10  # 10为满
        left_bottom_layout = QGridLayout()
        for i in range(2):  # 行
            for j in range(2):  # 列
                index = i * 2 + j
                # 创建一个 QVBoxLayout 为每个垃圾种类和其对应的图片
                garbage_layout = QVBoxLayout()
                garbage_text = QLabel(
                    f"{self.garbage_types[index]}: {self.numbers[index]}")
                garbage_text.setFont(left_bottom_font)
                garbage_text.setAlignment(Qt.AlignCenter)  # 居中对齐
                garbage_layout.addWidget(garbage_text)
                self.garbage_text_labels.append(garbage_text)  # 添加到列表中
                garbage_label = QLabel(self)
                pixmap = QPixmap(f"logo/{self.image_names[index]}")
                pixmap = pixmap.scaled(
                    width // 4, height // 4, Qt.KeepAspectRatio)
                garbage_label.setPixmap(pixmap)
                # garbage_label.setAlignment(Qt.AlignCenter)  # Center the image
                garbage_layout.addWidget(garbage_label)
                left_bottom_layout.addLayout(garbage_layout, i, j)
        left_bottom_widget = QWidget()
        left_bottom_widget.setLayout(left_bottom_layout)
        main_layout.addWidget(left_bottom_widget, 1, 0)  # 添加到主布局的左下角

        # 右下角：显示摄像头捕获的图像
        self.camera_image = None  # 初始化为 None
        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(width//2, height//2)
        main_layout.addWidget(self.camera_label, 1, 1)  # 添加到主布局的右下角

        # 设置主布局
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def update_full(self):
        print(f"满载信息:{self.full}")
        if self.full <= 3:
            print("进入update_full")
            top_right_font = QFont()
            top_right_font.setPointSize(40)  # 设置字体大小为16
            msg_box = QMessageBox(None)
            msg_box.setFont(top_right_font)
            msg_box.setWindowTitle("警告")  # 设置窗口标题
            msg_box.setText(f"{self.garbage_types[self.full]} 满载")
            msg_box.setIcon(QMessageBox.Warning)  # 设置图标为警告图标
            msg_box.setStandardButtons(QMessageBox.Ok)  # 添加一个确定按钮
            msg_box.resize(1200, 1000)
            # self.serial_thead0.stop()
            # msg_box.finished.connect(self.on_msg_box_closed)  # 连接到新的槽函数
            msg_box.exec_()
            print(f"检测到{self.garbage_types[self.full]}满载")
            self.full = 4  # 重置

    # def on_msg_box_closed(self):
    #     # self.serial_thead0.run()
    #     pass

    def update_video_slot(self, video_img):
        self.video_image = video_img
        v_height, v_width, _ = self.video_image.shape
        bytes_per_line = 3 * v_width
        q_img = QImage(self.video_image.data, v_width, v_height,
                       bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        v_pixmap = QPixmap.fromImage(q_img)
        v_pixmap = v_pixmap.scaled(
            self.width() // 2, self.height() // 2, Qt.KeepAspectRatio)
        self.video_label.setPixmap(v_pixmap)
        self.video_label.update()  # 强制更新QLabel

    def throwRubbish(self):
        self.insert_data_to_top("请投放垃圾")  # 添加新的文本到QTextEdit
        self.recent_data_label.update()  # 强制更新QLabel

    def update(self, frame, class_type):
        if frame is not None:  # 确保图像不是None
            height, width, _ = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height,
                           bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(
                600, 330, Qt.KeepAspectRatio)
            self.camera_label.setPixmap(pixmap)
            self.camera_label.update()  # 强制更新QLabel

        if class_type <= 3:
            self.numbers[class_type] = self.numbers[class_type]+1
            self.type = class_type
            print(f"垃圾总数 : {self.totalNumber}")
            print(f"垃圾数量 : {self.numbers}")
            # self.garbage_te
            for index, label in enumerate(self.garbage_text_labels):
                label.setText(
                    f"{self.garbage_types[index]}: {self.numbers[index]}")
                label.update()
            # self.totalNumber += ite('p'.encode())
            self.totalNumber += 1  # TODO:总数加1
        elif class_type == 4:  # 空白
            # self.serial_thead0.serial_0.write('m'.encode('ASCII'))
            # print("发送 m")
            pass

        if self.multi == False:
            # 单垃圾模式
            # self.totalNumber += 1
            pass
        # else:
            if (self.currentNum != 0) and (self.currentNum % global_vars.update_list[self.update_list_index] == 0):
                self.toUpdate = True
                self.update_list_index += 1
                self.update_times += 1

        if self.type not in [-1, 4]:
            if self.multi == True:
                # 多垃圾模式
                if self.toUpdate == False:
                    # 插入但不显示
                    self.currentNum += 1
                    print(f"垃圾总数:{self.totalNumber}")
                    if self.currentNum == 1:
                        self.new_data = f"{self.update_times} "
                    self.new_data += f"{self.garbage_types[self.type]} {self.numbers[self.type]} "
                    # 更新垃圾总数
                    self.QtotalNum.setText(f"垃圾总数:{self.totalNumber}")
                    self.QtotalNum.update()  # 强制更新QLabel
                    if (self.currentNum != 0) and ((self.currentNum) % (global_vars.update_list[self.update_list_index]-1) == 0):
                        self.toUpdate = True
                        self.update_list_index += 1
                        self.update_times += 1
                elif self.toUpdate == True:
                    # 多垃圾每次投放只算做一次totalNumber
                    # print("多垃圾:更新总垃圾数量")
                    # print(f"多垃圾 垃圾总数:{self.totalNumber}")
                    self.QtotalNum.setText(f"垃圾总数:{self.totalNumber}")
                    self.QtotalNum.update()  # 强制更新QLabel
                    # self.QtotalNum.repaint()
                    # self.new_data = f"{self.totalNumber} {self.garbage_types[self.type]} {self.numbers[self.type]} OK!\n"
                    self.new_data += f"{self.garbage_types[self.type]} {self.numbers[self.type]} "
                    self.new_data += f"OK!"
                    self.insert_data_to_top(self.new_data)
                    self.recent_data_label.update()  # 更新右上角的信息
                    # 重置
                    self.toUpdate = False
                    self.currentNum = 0
            else:
                # 单垃圾模式
                print("更新总垃圾数量")
                print(f"垃圾总数:{self.totalNumber}")
                self.QtotalNum.setText(f"垃圾总数:{self.totalNumber}")
                self.QtotalNum.repaint()
                # self.QtotalNum.update()  # 强制更新QLabel
                self.new_data = f"{self.totalNumber} {self.garbage_types[self.type]} {self.numbers[self.type]} OK!\n"
                self.insert_data_to_top(self.new_data)
                self.recent_data_label.repaint()
                # self.recent_data_label.update()  # 强制更新QLabel
                # 更新左下角的信息
                # self.QtotalNum.setText(f"垃圾总数:{self.totalNumber}")
                # self.QtotalNum.update()
                self.garbage_text_labels[self.type].setText(
                    f"{self.garbage_types[self.type]}: {self.numbers[self.type]}")
            print("更新当前信息")
        if self.totalNumber == 10:
            # self.serial_thread0.serial_0.write('p'.encode())
            print("发送 p")

    def start_inference(self):
        singal = self.serial_thread0.get_singal()
        print(f"start inference singal {singal}")
        if singal == 'a':
            self.infer_thread.startExecutionSignal.emit()

    def start_inference_a(self):
        self.infer_thread.infer_a()

    def start_inference_t(self, signal):
        # print(f'从port1接受到{signal}')
        # signal='t'
        # if signal == 't':
        #     print("从serial 1 进入推理")
        #     self.infer_thread.startExecutionSignal.emit()
        #     signal=''
        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:  # 检查是否是 Esc 键

            self.close()  # 关闭窗口
        elif event.key() == Qt.Key_I:
            print("按下 I 进入推理")
            self.infer_thread.startExecutionSignal.emit()  # 按I推理

    def serial_signal0(self, signal):
        # 处理从port0的信号
        # print(f"从port0接受到{signal}")
        if signal == 'T':
            self.throwRubbish()
        # if signal in ['w','y']:
        #     self.serial_signal_up_raster=signal
        elif signal == 'K':
            print(f'receve recyclable_signal')
            print("可回收垃圾满载")
            self.full = 0
            self.update_full()
        elif signal == 'C':
            self.full = 1
            print(f'receve foodScrap_signal')
            print("厨余垃圾满载")
            self.update_full()
        elif signal == 'Y':
            self.full = 2
            print(f'receve hazardous_pin')
            print("有害垃圾满载")
            self.update_full()
        elif signal == 'Q':
            self.full = 3
            print(f'receve others_signal')
            print("其他垃圾满载")
            self.update_full()
        elif signal == 't':
            print("从serial 0 接受到信号 t 进入推理")
            self.infer_thread.startExecutionSignal.emit()

    def insert_data_to_top(self, new_data):
        # 把新的数据放在顶部
        current_data = self.recent_data_label.toPlainText()
        updated_data = new_data + "\n" + current_data
        self.recent_data_label.setPlainText(updated_data)
