import cv2
import pandas as pd
import cvzone
import numpy as np
from ultralytics import YOLO
from tracker import *

model = YOLO('yolov8s.pt') # affect to accuracy of object when tracking

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

tracker = Tracker()

while True:
  success,frame = cap.read()
  if not success:
    break

  frame = cv2.resize(frame,(1020,500))

  results = model.predict(frame)
  result = results[0].boxes.data
  data_result = pd.DataFrame(result).astype('float')

  list = []

  for i,row in data_result.iterrows():
    x1 = int(row[0])
    y1 = int(row[1])
    x2 = int(row[2])
    y2 = int(row[3])

    class_object = int(row[5])
    object_class = class_list[class_object]

    if 'person' in object_class: # only person class object
      list.append([x1, y1, x2, y2])

    # detect with bbox
    bbox_idx = tracker.update(list)
    for bbox in bbox_idx:
      x3,y3,x4,y4,id = bbox
      cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
      cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)

  print(list)
  cv2.imshow('RGB', frame)
  if cv2.waitKey(1) & 0xFF==27:
    break

cap.release()
cv2.destroyAllWindows()