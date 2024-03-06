import cv2 as cv
import numpy as np
from PIL import Image
import os
path = "dataset"

recognizer = cv.face.LBPHFaceRecognizer_create()
detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def getImagesLabels(path):
    imgpaths = [os.path.join(path.f) for f in os.listdir(path)]
    samples = []
    ids = []
    for imgpath in imgpaths:
        # PIL_img = Image.open(imagePath), convert('L')
        img = cv.imread('imagePath', cv.IMREAD_GRAYSCALE)
        img_numpy = np.array(img, 'uint8')
        
        id = int(os.path.split(imgpath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        
        for (x,y,w,h) in faces:
            samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return samples, ids

faces, ids = getImagesLabels(path)
recognizer.train(faces, ids)
recognizer.write("train/train.yml")
print("\n {0} khuon mat duoc train".format(len(np.unique(ids))))
        
         
