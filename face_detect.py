import cv2 as cv
import os

cam = cv.VideoCapture(0)
cam.set(4, 480)
face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = input('\n Nhap id khuon mat: ')
print('\n Khoi tao camera ...')
count = 0
while(True):
    ret, img = cam.read()
    img = cv.flip(img, -1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        count += 1
        cv.imwwrite("dataset/user."+ str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv.imshow('image', img)
    if cv.waitKey(100) & 0xff == ord('q'):
        break
    elif count >= 30:
        break
cam.release()
cv.destroyAllWindows()
        