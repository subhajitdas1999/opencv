import os
import cv2
import numpy as np 
from PIL import Image
import pickle


face_cascade=cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()

BASE_DIR=os.path.dirname(os.path.abspath(__file__))


Image_dir=os.path.join(BASE_DIR,"images")

labels_id={}
x_train=[]
y_lable_id=[]
count=0
num=1

cap=cv2.VideoCapture(0)
name="subhajit"
def taking_img(name):
	if os.path.exists("images/"+name):
		print("person already exists")
	else:
		os.mkdir("images/"+name)
		while num<=100:
			ret,frame=cap.read()
			gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			cv2.imshow("video",frame)
			cv2.imwrite("images/"+name+"/user."+("faceme%d.png"%(num)),gray)
			num+=1
			if cv2.waitKey(20) & 0xFF==ord('x'):
				break

	cap.release()
	cv2.destroyAllWindows()

taking_img(name)

for root,dirs,files in os.walk(Image_dir):
	for file in files:
		if file.endswith(".jpg") or file.endswith(".png"):
			path=os.path.join(root,file)
			lable=os.path.basename(root)
			if lable not in labels_id:
				labels_id[lable]=count
				count+=1
			id_=labels_id[lable]
			pil_img=Image.open(path).convert('L')
			Image_array=np.array(pil_img,"uint8")
			faces=face_cascade.detectMultiScale(Image_array,scaleFactor=1.5,minNeighbors=5)
			for (x,y,w,h) in faces:
				roi=Image_array[y:y+h,x:x+w]
				x_train.append(roi)
				y_lable_id.append(id_)

#print(labels_id)
with open("my_lable.pkl","wb") as obj:
	pickle.dump(labels_id,obj)


recognizer.train(x_train,np.array(y_lable_id))
recognizer.save("my_traning.yml")






