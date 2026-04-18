from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout
import serial
from PyQt5.QtCore import QThread, pyqtSignal
import global_vars

class Serial_thread(QThread):
    serial_signal0 = pyqtSignal(str)
    def __init__(self,port, muti=False):
        super().__init__()
        print("创建serial线程")
        self.serial_0 = serial.serialposix.Serial(
            port, 9600, timeout=1000)
        self.singal0: str = ''
        self.is_running = False
    def get_singal(self):
        return self.singal0
    def run(self):
        # self.is_running = True
        # while self.is_running:
        while True:
            # i+=1
            self.singal0 = self.serial_0.read().decode('ASCII')
            # print(f"正在接受信号 singal0:{self.singal0}")
            if self.singal0 in global_vars.singal0_list:
                # print(f'port0接受信号到{self.singal0}')
                self.serial_signal0.emit(self.singal0)
                # print(f'port1发送信号{self.singal0}')
                self.singal0=''#重置      
    def stop(self):
        self.is_running = False 