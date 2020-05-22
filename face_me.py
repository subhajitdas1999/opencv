import numpy as np 
import cv2
import pickle

face_cascades=cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("my_traning.yml")

lable={}
with open("my_lable.pkl","rb") as obj:
	new_lable=pickle.load(obj)
	lable={v:k for k,v in new_lable.items()}




cap=cv2.VideoCapture(0)

while True:
	ret,frame=cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=face_cascades.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
	for(x,y,w,h) in faces:
		roi_gray=gray[y:y+h,x:x+w]
		roi_color=frame[y:y+h,x:x+w]
		id_,conf=recognizer.predict(roi_gray)
		if conf>=45 and conf <=70:
			font=cv2.FONT_HERSHEY_SIMPLEX
			color=(255,0,0)
			strock=2
			name=lable[id_]
			cv2.putText(frame,name,(x,y),font,1,color,strock,cv2.LINE_AA)
		color=(255,0,255)
		strock=2
		cv2.rectangle(frame,(x,y),(x+w,y+h),color,strock)
	cv2.imshow("video",frame)
	if cv2.waitKey(20) & 0xFF==ord('x'):
		break
cap.release()
cv2.destroyAllWindows()