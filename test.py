import os

# 文件夹路径
folder_path = '/home/nano/lobe/data'

# 初始化一个空列表，用于存储子文件夹名称
subfolders = []

# 遍历文件夹
for entry in os.listdir(folder_path):
    full_path = os.path.join(folder_path, entry)
    # 检查这个路径是否是一个目录
    if os.path.isdir(full_path):
        subfolders.append(entry)

subfolders=sorted(subfolders)
print(subfolders)
# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout
# from PyQt5.QtCore import QTimer
# import serial
# import global_vars
# serial_0 = serial.serialposix.Serial(
#             global_vars.port0, 9600, timeout=1.5)
# i=0
# while True:
#     i+=1
#     singal0 = serial_0.read().decode('ASCII')
#     print(f"\033[0m正在接受信号 singal0:{singal0} 次数{i}")
#     if singal0 in global_vars.singal0_list:
#         print("接受到了 \033[91m {serial0}")
#         singal0=''#重置      