from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
import global_vars
import time
# import Jetson.GPIO as GPIO


class GPIOThread(QThread):
    gpio_signal = pyqtSignal(int)

    def __init__(self, recyclable_pin, foodScrap_pin, hazardous_pin, others_pin):
        super().__init__()
        self.recyclable_pin = recyclable_pin
        self.foodScrap_pin = foodScrap_pin
        self.hazardous_pin = hazardous_pin
        self.others_pin = others_pin
        GPIO.setup(recyclable_pin, GPIO.IN)
        GPIO.setup(foodScrap_pin, GPIO.IN)
        GPIO.setup(hazardous_pin, GPIO.IN)
        GPIO.setup(others_pin, GPIO.IN)
        self.is_running = False
        self.full_type = 10  # 10代表没有满
        # 暂停线程
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.paused = False

    def run(self):
        self.is_running = True
        while self.is_running:
            self.mutex.lock()
            if self.paused:
                self.condition.wait(self.mutex)
            self.mutex.unlock()
            self.on_gpio_detected()
            self.gpio_signal.emit(self.full_type)
            time.sleep(0.2)  # 根据需要调整

    def on_gpio_detected(self):
        global_vars.recyclable_signal = GPIO.input(self.recyclable_pin)
        global_vars.foodScrap_signal = GPIO.input(self.foodScrap_pin)
        global_vars.hazardous_signal = GPIO.input(self.hazardous_pin)
        global_vars.others_signal = GPIO.input(self.others_pin)
        print(f"\033[92m第一次:{global_vars.recyclable_signal}  {global_vars.foodScrap_signal}  {global_vars.hazardous_signal} {global_vars.others_signal}\033[92m")
        time.sleep(0.5)
        global_vars.recyclable_signal1 = GPIO.input(self.recyclable_pin)
        global_vars.foodScrap_signal1 = GPIO.input(self.foodScrap_pin)
        global_vars.hazardous_signal1 = GPIO.input(self.hazardous_pin)
        global_vars.others_signal1 = GPIO.input(self.others_pin)
        print(f"\033[93m第二次:{global_vars.recyclable_signal1}  {global_vars.foodScrap_signal1}  {global_vars.hazardous_signal1} {global_vars.others_signal1}\033[93m")
        if global_vars.recyclable_signal == 0 and global_vars.recyclable_signal1 == 0:
            print(f'receve recyclable_signal')
            print("可回收垃圾满载")
            self.full_type = 0
        elif global_vars.foodScrap_signal == 0 and global_vars.foodScrap_signal1 == 0:
            print("厨余垃圾满载")
            print(f'receve foodScrap_signal')
            self.full_type = 1
        elif global_vars.hazardous_signal == 0 and global_vars.hazardous_signal1 == 0:
            print(f'receve hazardous_pin')
            print("有害垃圾满载")
            self.full_type = 2
        elif global_vars.others_signal == 0 and global_vars.others_signal1 == 0:
            print(f'receve others_signal')
            print("其他垃圾满载")
            self.full_type = 3
        else:
            self.full_type = 10

    def pause(self):
        # 暂停线程
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()

    def resume(self):
        # 恢复线程
        self.mutex.lock()
        self.paused = False
        self.condition.wakeAll()
        self.mutex.unlock()

    def stop(self):
        # 停止线程
        self.is_running = False
