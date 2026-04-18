from PyQt5.QtCore import QThread, pyqtSignal
import global_vars
import time
import cv2
import numpy as np


class VideoThread(QThread):
    video_signal = pyqtSignal(np.ndarray)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.video = cv2.VideoCapture(self.video_path)

    def run(self):
        while not global_vars.video_stop:
            ret, frame = self.video.read()
            if not ret:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self.video_signal.emit(frame)
            time.sleep(0.01)  # 根据视频的帧率调整
