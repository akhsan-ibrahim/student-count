import cv2
import pandas as pd
import cvzone
import numpy as np
from ultralytics import YOLO
from tracker import *

model = YOLO('yolov8n.pt')

# cursor coordinate
def RGB(event,x,y,flags,param):
  if event == cv2.EVENT_MOUSEMOVE:
    point = [x,y]
    print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB',RGB)

cap = cv2.VideoCapture('students.mp4')

# open the file that contains the list of object classes
my_file = open('coco.txt','r')
data = my_file.read()
class_list = data.split('\n')

while True:
  success,frame = cap.read()
  if not success:
    break

  frame = cv2.resize(frame,(1020,500))

  results = model.predict(frame)
  result = results[0].boxes.data
  data_result = pd.DataFrame(result).astype('float')

  for i,row in data_result.iterrows():
    x1 = int(row[0])
    y1 = int(row[1])
    x2 = int(row[2])
    y2 = int(row[3])

    class_object = int(row[5])
    object_class = class_list[class_object]

    if 'person' in object_class:
      cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)
      cvzone.putTextRect(frame,f'{object_class}',(x1,y1),1,1)


  cv2.imshow('RGB', frame)
  if cv2.waitKey(1) & 0xFF==27:
    break

cap.release()
cv2.destroyAllWindows()