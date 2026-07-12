#2026-07-12 17:24

'''
action/state + Output Log
事件(event)驱动 日志保存
'''
#-----------------------------------#
import cv2
from ultralytics import YOLO
from datetime import datetime
#-----------------------------------#
ROI_LINE_X=540
MIN_CONF=0.25
TARGET_CLASS="person"
LOG_FILE="log_debug/event.log"
#-----------------------------------#
def triggered_warning():#解耦
    print(f"[{datetime.now():%H:%M:%S}] Warning: person entered ROI.") 

def generate_ssname():#解耦
    return datetime.now().strftime("%Y%m%d_%H%M%S")+".jpg" 

def append_log(ss_name):#追加编辑专用日志
    now = datetime.now()
    with open(LOG_FILE,'a',encoding="utf-8") as f:
        f.write(
            f"[{now:%Y-%m-%d %H:%M:%S}]"
            f"Warning Person intruded ROI\n"
        )
        f.write(
            f"Screenshot: {ss_name}\n"
        )
        f.write(
            "-"*40+"\n"   #换行标记
        )
#-----------------------------------#

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
            triggered_warning()
            ss_name=generate_ssname()
            #动作:追加记入日志
            append_log(ss_name)
        last_warning=warning; #fix:一旦退出再进入需要重新报警 

    cv2.imshow("cam",frame)
    if(cv2.waitKey(1)&0xff==ord('q')): break

cap.release()
cv2.destroyAllWindows() 
#-----------------------------------#