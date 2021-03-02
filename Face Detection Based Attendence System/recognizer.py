import cv2, numpy as np;
import xlwrite
import time
import sys
from playsound import playsound
start=time.time()
period=8
face_cas = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW);
recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.read("Trainner.yml");
flag = 0;
id=0;
filename='filename';
dict = {
            'item1': 1
}
#font = cv2.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 5, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = cap.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cas.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    c=0
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        te = f"{id}: {conf}"
        cv2.putText(img, te,(x,y-10),font,0.55,(120,255,120),1)
        if(conf < 50):
            if(id==1):
                id='Animesh Mondal'
                if((str(id)) not in dict):
                    filename=xlwrite.output('attendance','class1',1,id,'yes')
                    dict[str(id)]=str(id)
            elif(id==2):
                id = 'Vivek Kishore'
                if ((str(id)) not in dict):
                    filename =xlwrite.output('attendance', 'class1', 2, id, 'yes')
                    dict[str(id)] = str(id)
            elif(id==3):
                id = 'Debjit Kar'
                if ((str(id)) not in dict):
                    filename =xlwrite.output('attendance', 'class1', 3, id, 'yes')
                    dict[str(id)] = str(id)
            elif(id==4):
                id = 'Kunal Pathak'
                if ((str(id)) not in dict):
                    filename =xlwrite.output('attendance', 'class1', 4, id, 'yes')
                    dict[str(id)] = str(id)
            elif(id==5):
                id = 'Subhradip Barik'
                if ((str(id)) not in dict):
                    filename =xlwrite.output('attendance', 'class1', 5, id, 'yes')
                    dict[str(id)] = str(id)
            else:
                if ((str(id)) not in dict):
                    filename =xlwrite.output('attendance', 'class1', id , id, 'yes')
                    dict[str(id)] = str(id)
                    flag=flag+1
                break

        #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,(0,0,255));
    cv2.imshow('frame',img)
    #cv2.imshow('gray',gray);
    if flag == 10:
        playsound('transactionSound.mp3')
        print('transactionSound.mp3')
        break
    if time.time()>start+period:
        break

    if cv2.waitKey(100) == 100  & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()
