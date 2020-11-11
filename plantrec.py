import numpy as np 
import cv2

#load custom classifier
#cascade trainer - https://amin-ahmadi.com/cascade-trainer-gui/
#sample folder must contain positive that plant is in frame and negative plant not in frame
#positive image usage percentage and negative image count needed as well
#https://identify.plantnet.org/ for plant datasets

plant_cascade = cv2.CascadeClassifier('SOURCE')
cap = cv2.VideoCapture(0)

while 1: 
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plant = plant_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in plant:
            cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

    cv2.imshow('img,' img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cv2.destoryAllWindows()