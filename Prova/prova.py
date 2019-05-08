# -*- coding: UTF-8 -*-


import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (224, 224)

face_cascade = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')



video = cv2.VideoCapture('video/video3.mp4')

size = (
  int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
  int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

codec = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('video/Saida.mp4', codec, 23.0, size)


while(True):
    ret, frame = video.read()
    if frame is None:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    output.write(frame)

video.release()
cv2.waitKey(0)
cv2.destroyAllWindows()