import cv2
import os
import numpy as np
from PIL import Image 
import pickle
import sqlite3
import urllib

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:/AliRiza_Scripts_Python/Face_Recognition-Sqlite/recognizer/trainningData.yml")
cascadePath = "C:/AliRiza_Scripts_Python/Face_Recognition-Sqlite/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = "C:/AliRiza_Scripts_Python/Face_Recognition-Sqlite/dataSet"

def getProfile(id):
    conn = sqlite3.connect("C:\AliRiza_Scripts_Python\Face_Recognition-Sqlite\FaceBase.db")
    cmd= "SELECT * FROM people WHERE ID="+str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile




font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)

url = "http://192.168.2.115:8080/shot.jpg"


while True:
    ret, img = cam.read();
    imgResp = urllib.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()),dtype = np.uint8)
    img = cv2.imdecode(imgNp,-1)
    if ord("q")==cv2.waitKey(10):
        exit()
    gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        profile = getProfile(id)
        if(profile!=None):
             cv2.putText(img,"Ad: " + str(profile[1]),(x,y+h+30), font, 1, (0,0,255), 2, cv2.LINE_AA)
             cv2.putText(img,"Yas: " + str(profile[2]),(x,y+h+60), font, 1, (0,0,255), 2, cv2.LINE_AA)
             cv2.putText(img,"Cinsiyet: " + str(profile[3]),(x,y+h+90), font, 1, (0,0,255), 2, cv2.LINE_AA)
             cv2.putText(img,"suc: " + str(profile[4]),(x,y+h+120), font, 1, (0,0,255), 2, cv2.LINE_AA)
             
    cv2.imshow('test',img)
    if(cv2.waitKey(1)==ord("q")):
        break;
cam.release()
cv2.destroyAllWindows
