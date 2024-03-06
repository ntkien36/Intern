import cv2 as cv
import numpy as np
import os

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('train/train.yml')
faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv.FONT_HERSHEY_SIMPLEX

id = 0
names = ['1', '2']
cam = cv.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minw = 0.1 * cam.get(3)
minh = 0.1 * cam.get(4)

while True:
    
    ret, img = cam.read()
    img = cv.flip(img, -1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray, scaleFactor= 1.2, minNeighbors= 5, minSize= (int(minw), int(minh)))
    for(x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), (255, 0,0), 2)
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if (conf < 100):
            id = names[id]
            cof = " {0}%".format(round(100 - conf))
        else:
            id = "Unknown"
            conf = " {0}%".format(round(100 - conf))
        cv.putText(img, str(id), (x+5, y-5), font, 1, (255,255,255), 2)
        cv.putText(img, str(conf), (x+5, y+h-5), font, 1, (255,255,), 2)
        
    cv.imshow(img)
    if cv.waitKey(100) & 0xff == ord('q'):
        break
cam.release()
cv.destroyAllWindows()
