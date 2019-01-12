from PIL import Image
import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create();
path = "C:\AliRiza_Scripts_Python\Face_Recognition-Sqlite\dataSet"

def getImageWithID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert("L");
        faceNp = np.array(faceImg,"uint8")
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(faceNp)
        print ID
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return IDs, faces

Ids,faces = getImageWithID(path)
recognizer.train(faces,np.array(Ids))
recognizer.save("C:/AliRiza_Scripts_Python/Face_Recognition-Sqlite/recognizer/trainningData.yml")
cv2.destroyAllWindows()
