#2026-07-03 21:55

'''
predict()
模型推测 - video
'''

from ultralytics import YOLO

model = YOLO("model/yolo11n.pt")

results = model.predict(
    "videos/IMG_6615.MOV",
    save=True
)

print(results)