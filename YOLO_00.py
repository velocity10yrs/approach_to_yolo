#2026-06-30 15:37

'''
enviroment set up
开发环境配置
'''

from ultralytics import YOLO
import ultralytics
import torch
import cv2

print("Ultralytics :", ultralytics.__version__) # Ultralytics : 8.4.83
print("Torch        :", torch.__version__) # Torch        : 2.2.2
print("OpenCV       :", cv2.__version__) # OpenCV       : 4.10.0
print("CUDA         :", torch.cuda.is_available()) # CUDA         : False
