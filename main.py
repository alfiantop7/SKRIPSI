import common as cm
import cv2
import numpy as np
from PIL import Image
import time
from threading import Thread

import sys
sys.path.insert(0, '/var/www/html/earthrover')

cap = cv2.VideoCapture(1000)
threshold=0.2
top_k=5 #number of objects to be shown as detected

#model directory
model_dir = '/var/www/html/all_models'
model= 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
lbl = 'coco_labels.txt'

import time
def time_elapsed(start_time,event):
        time_now=time.time()
        duration = (time_now - start_time)*1000
        duration=round(duration,2)
        print (">>> ", duration, " ms (" ,event, ")")

tolerance=0.1
x_deviation=0
y_max=0

object_to_track='person'

def track_object(objs,labels):
    
    #global delay
    global x_deviation, y_max, tolerance
    
    if(len(objs)==0):
        print("no objects to track")
        return

    flag=0
    for obj in objs:
        lbl=labels.get(obj.id, obj.id)
        if (lbl==object_to_track):
            x_min, y_min, x_max, y_max = list(obj.bbox)
            flag=1
            break
        
    #print(x_min, y_min, x_max, y_max)
    if(flag==0):
        print("selected object no present")
        return
        
    x_diff=x_max-x_min
    y_diff=y_max-y_min
         
    obj_x_center=x_min+(x_diff/2)
    obj_x_center=round(obj_x_center,3)
    
    obj_y_center=y_min+(y_diff/2)
    obj_y_center=round(obj_y_center,3)
    
    x_deviation=round(0.5-obj_x_center,3)
    y_max=round(y_max,3)
        
    print("{",x_deviation,y_max,"}")
   
    thread = Thread(target = move_robot)
    thread.start()

def move_robot():
    global x_deviation, y_max, tolerance
    
    y=1-y_max #distance from bottom of the frame
    
    if(abs(x_deviation)<tolerance):
        if(y<0.1):
            print("reached person")
    
        else:
            print("moving forward")
    
    
    else:
        if(x_deviation>=tolerance):
            delay1=get_delay(x_deviation)
                
            time.sleep(delay1)
            print("moving left")
    
                
        if(x_deviation<=-1*tolerance):
            delay1=get_delay(x_deviation)
                
            time.sleep(delay1)
            print("moving right")

def get_delay(deviation):
    
    deviation=abs(deviation)
    
    if(deviation>=0.4):
        d=0.080
    elif(deviation>=0.35 and deviation<0.40):
        d=0.060
    elif(deviation>=0.20 and deviation<0.35):
        d=0.050
    else:
        d=0.040
    
    return d

def main():
  
    interpreter, labels = cm.load_model(model_dir,model,lbl)
    
    fps=1
   
    while True:
        start_time=time.time()
        
        #----------------Capture Camera Frame-----------------
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2_im = frame
        cv2_im = cv2.flip(cv2_im, 0)
        cv2_im = cv2.flip(cv2_im, 1)

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
       
        #-------------------Inference---------------------------------
        cm.set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = cm.get_output(interpreter, score_threshold=threshold, top_k=top_k)
        
        #-----------------other------------------------------------
        track_object(objs,labels)#tracking  <<<<<<<
       
        fps = round(1.0 / (time.time() - start_time),1)
        print("*********FPS: ",fps,"************")

    cap.release()
    cv2.destroyAllWindows()

main()

