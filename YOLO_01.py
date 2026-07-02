#2026-07-01 11:32

'''
get to know about model 
观察模型
'''

from ultralytics import YOLO

#model = YOLO("yolo11x.pt") 大分量模型 不太适合个人电脑
#model = YOLO("yolo11l.pt") 大分量模型 不太适合个人电脑
#model = YOLO("yolo11m.pt") 大分量模型 不太适合个人电脑
#model = YOLO("yolo11s.pt") 大分量模型 不太适合个人电脑
model = YOLO("model/yolo11n.pt") 
print(model)