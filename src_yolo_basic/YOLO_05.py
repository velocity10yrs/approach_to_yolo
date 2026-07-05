#2026-07-05 11:57

'''
results=predict()
解读results属性
'''

import cv2 
from ultralytics import YOLO
import time

model=YOLO("models/yolo11n.pt")
cap=cv2.VideoCapture(0)

result = None

while True:
    ret,frame=cap.read()
    if not ret: break
    
    start=time.time()#返回单位为 秒second 
    
    results=model.predict(frame,conf=0.25,verbose=False)#verbose关闭控制台输出
    result_frame=results[0].plot()
    end=time.time()
    fps=1/(end-start)#计算帧

    cv2.putText(result_frame,f"FPS:{fps:.2f}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,2))#实时FPS
    cv2.imshow("yolo cam",result_frame)
    if(cv2.waitKey(1)&0xFF==ord('q')): break

    result=results[0] #抓最后一帧

cap.release()
cv2.destroyAllWindows()

print(type(result))
print("*********************************")
print(result.boxes)
print("*********************************")
print(result.boxes.xyxy)
print("*********************************")
print(result.boxes.cls)
print("*********************************")
cls=int(result.boxes.cls[0])
print(result.names[cls])
print("*********************************")
print(result.boxes.conf)
print("*********************************")