#2026-07-07 13:13

'''
ROI (region of interest)
划分危险区域
'''
ROI_LINE_X=540 #fix:ROI_LINE方向不明

import cv2
from ultralytics import YOLO

model=YOLO("models/yolo11n.pt")
cap=cv2.VideoCapture(0)
#**************************************#
while True:
    #section-1:frame接收
    ret,frame=cap.read()
    if not ret: break
    #section-1.2*:frame处理
    #section-2.1:frame预测
    results=model.predict(frame,conf=0.25,verbose=False)
    res=results[0]
    #section-2.2:目标追踪
    #opt: 线放到循环外只画一次
    cv2.line(frame,                     
             (ROI_LINE_X,0),
             (ROI_LINE_X,frame.shape[0]), #bug:frame.shape[1]
             (0,0,255),
             2)
    #section-3:目标筛选
    for box in res.boxes:
        cls=int(box.cls)
        name=res.names[cls]
        if(name!='person'): continue
    #section-4: ROI(危险区域)划出
        x1,y1,x2,y2=map(int,box.xyxy[0])
        centre_x=(x1+x2)//2
        centre_y=(y1+y2)//2
        cv2.circle(frame,(centre_x,centre_y),5,(255,0,0),-1)
    #section-5:目标标出
        if(centre_x<ROI_LINE_X):
            cv2.putText(frame,"warning",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow("cam",frame)
    if(cv2.waitKey(1)&0xff==ord('q')): break
#**************************************#
cap.release()
cv2.destroyAllWindows()