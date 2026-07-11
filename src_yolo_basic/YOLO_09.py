#2026-07-11 12:56

'''
action/state + Output Screenshot
事件(event)驱动 截图保存
'''

import cv2
from ultralytics import YOLO
from datetime import datetime
import os

ROI_LINE_X=540
MIN_CONF=0.25
TARGET_CLASS="person"
SAVE_DIR = "output_3/screenshot/"
#filename = ""
os.makedirs(SAVE_DIR,exist_ok=True)

def triggered_warning(): #警报func()
    now = datetime.now()
    print(
        f"[{now:%H:%M:%S}] Warning: person entered ROI."
    ) 
    filename = now.strftime("%Y%m%d_%H%M%S")+".jpg" #fix:别忘了后缀格式
    return filename

def triggered_ss(filename,frame): #截图保存func()
    filepath = os.path.join(SAVE_DIR,filename)
    ok = cv2.imwrite(filepath,frame)
    if ok: print(f"ss saved:{filepath}")

cap=cv2.VideoCapture(0)
model=YOLO("model/yolo11n.pt")
last_warning = False

while True:
    #读帧
    ret,frame=cap.read()
    if not ret: break
    #预测
    results=model.predict(frame,conf=MIN_CONF,verbose=False)
    res=results[0]
    #ROI
    cv2.line(frame,
             (ROI_LINE_X,0),(ROI_LINE_X,frame.shape[0]),
             (0,0,255),2)
    #筛选
    for box in res.boxes:
        cls=int(box.cls)
        name=res.names[cls] #fix
        if(name!=TARGET_CLASS): continue
        #
        x1,y1,x2,y2=map(int,box.xyxy[0])
        cen_x=(x1+x2)//2;cen_y=(y1+y2)//2
        cv2.circle(frame,
                   (cen_x,cen_y),
                   5,(255,0,0),-1)
        #事件:警报
        warning = (cen_x<=ROI_LINE_X)
            #last_warning = False #opt
        #动作:警报驱动
        if warning and not last_warning:
            last_warning=warning #只报警新目标
            cv2.rectangle(frame,
                          (x1,y1),(x2,y2),
                          (255,0,0),2)
            cv2.putText(frame,"WARNING",
                        (30,50),cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,0,255),3)
            filename=triggered_warning()
            #动作:截图保存
            triggered_ss(filename,frame)
        last_warning=warning; #fix:一旦退出再进入需要重新报警 

    cv2.imshow("cam",frame)
    if(cv2.waitKey(1)&0xff==ord('q')): break

cap.release()
cv2.destroyAllWindows() 