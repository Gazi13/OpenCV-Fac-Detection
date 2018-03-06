import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
# you can give fullpath
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')



if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')

##by Camera or Video 
cap = cv2.VideoCapture(0)
##cap = cv2.VideoCapture('VideoPath.mp4')


while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        mouth_rects = mouth_cascade.detectMultiScale(gray[y:y+h, x:x+w], 1.3, 5)      
        for (mx,my,mw,mh) in mouth_rects:
            cv2.rectangle(roi_color, (mx,my), (mx+mw,my+mh), (0,0,255), 2)

##        nose = nose_cascade.detectMultiScale(gray[y:y+h, x:x+w], 1.3, 3)      
##        for (nx,ny,nw,nh) in nose:
##            cv2.rectangle(roi_color, (nx,ny/2), (nx+nw,ny+nh), (150,150,150), 2)    
            
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
