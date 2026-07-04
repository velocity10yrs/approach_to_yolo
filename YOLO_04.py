#2026-07-04 14:04

'''
predict()
模型realtime推测 - camera
'''

import cv2 
from ultralytics import YOLO
import time

model=YOLO("models/yolo11n.pt")
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    if not ret: break
    
    start=time.time()#返回单位为 秒second 
    
    #对照-1:直接处理原帧
    results=model.predict(frame,conf=0.25,verbose=False)#verbose关闭控制台输出
    #对照-2:处理灰度帧
    #frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#原帧转灰度
    #results=model.predict(frame_gray,conf=0.25,verbose=False)
    result_frame=results[0].plot()
    end=time.time()
    fps=1/(end-start)#计算帧

    cv2.putText(result_frame,f"FPS:{fps:.2f}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,2))#实时FPS
    cv2.imshow("yolo cam",result_frame)
    if(cv2.waitKey(1)&0xFF==ord('q')): break

cap.release()
cv2.destroyAllWindows()