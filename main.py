import cv2
import pandas as pd
import cvzone
import numpy as np
from ultralytics import YOLO
from tracker import *

model = YOLO('yolov8m.pt') # affect to accuracy of object when tracking

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

tracker = Tracker() # define object tracker

area1 = [(494,289),(505,499),(578,496),(530,292)]
area2 = [(548,290),(600,496),(637,493),(574,288)]

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
      # distance between object point and object polygon
      result = cv2.pointPolygonTest(
        np.array(area1,np.int32), # polygon area
        ((x4,y4)), # object point
        False # measure distance with polygon
      )
      # when object point on the polygon
      if result >= 0:
        cv2.circle(frame,(x4,y4),7,(255,0,255),-1) # show object point
        cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2) # show object bbox
        cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1) # show object label (id)

  # print(list)

  # set detection area
  cv2.polylines(
    frame, # drawing medium
    [np.array(area1,np.int32)], # change format fit to polylines format
    True, # closed polygon --> every poin is connected
    (0,255,0), # RGB color
    2 # line weight
  )
  cv2.imshow('RGB', frame)
  if cv2.waitKey(1) & 0xFF==27:
    break

cap.release()
cv2.destroyAllWindows()