#2026-07-02 19:08

'''
predict()
模型推测
'''

from ultralytics import YOLO
#loading model
model=YOLO("model/yolo11n.pt")
#推理
results = model.predict(
    source="images/(meme) 睡眠不足.jpeg",
    save=True,
    conf=0.25
)

print(results)
