# 2026-07-06 10:02

'''
result=predict()
解构预测结果+编写业务逻辑
'''

from ultralytics import YOLO
import cv2 

model=YOLO("model/yolo11n.pt")
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    if not ret: break

    results=model.predict(frame,conf=0.25,verbose=False)
    res=results[0]

    for box in res.boxes:
        cls=int(box.cls)
        name=res.names[cls]
        #过滤目标
        if(name=='person'):
            #print(name)
            #筛选目标
            conf = float(box.conf)
            if(conf<0.8):continue
            #矩形标出
            x1,y1,x2,y2=map(int,box.xyxy[0]) #左上角and右下角
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,'person',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        else:
            continue
    
    cv2.imshow("yolo_cam",frame)
    if(cv2.waitKey(1)&0xff==ord('q')): break

cap.release()
cv2.destroyAllWindows()
        