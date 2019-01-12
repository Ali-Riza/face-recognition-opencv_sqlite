import cv2
import numpy as np
import sqlite3
import urllib

faceDetect=cv2.CascadeClassifier("C:\AliRiza_Scripts_Python\Face_Recognition-Sqlite\haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0);

def insertOrUpdate(Id,Name):
    conn = sqlite3.connect("C:\AliRiza_Scripts_Python\Face_Recognition-Sqlite\FaceBase.db")
    cmd ="SELECT * FROM people WHERE ID="+str(Id)
    cursor = conn.execute(cmd)
    IsRecordExist = 0
    for row in cursor:
        IsRecordExist = 1
    if (IsRecordExist == 1):
        cmd="UPDATE people SET NAME="+str(Name)+" WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO People(ID,Name) Values("+str(Id)+","+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()
        
id = raw_input("enter user id ")
name = raw_input("enter your name ")
insertOrUpdate(id,name)
sampleNum = 0;

url = "http://192.168.2.115:8080/shot.jpg"

while(True):
    ret,img = cam.read();
    ret, img = cam.read();
    imgResp = urllib.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()),dtype = np.uint8)
    img = cv2.imdecode(imgNp,-1)
    if ord("q")==cv2.waitKey(10):
        exit()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        sampleNum = sampleNum +1;
        cv2.imwrite("C:/AliRiza_Scripts_Python/Face_Recognition-Sqlite/dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,225),2)
        cv2.waitKey(100)
    cv2.imshow("Face",img);
    cv2.waitKey(1)
    if(sampleNum>300):
        break
cam.release()
cv2.destroyAllWindows()


    
