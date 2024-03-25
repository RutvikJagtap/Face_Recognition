# Creating database 
import cv2, sys, numpy, os 
from time import sleep
haar_file = 'haarcascade_frontalface_default.xml'

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

datasets = r"dataset"
name = input("Enter your value: ")

path = os.path.join(datasets, name) 
if not os.path.isdir(path): 
	os.mkdir(path)
(width, height) = (130, 100)	 

face_cascade = cv2.CascadeClassifier(haar_model) 
webcam = cv2.VideoCapture(0) 
count = 1
while count < 50: 
	(_, im) = webcam.read() 
	faces = face_cascade.detectMultiScale(im, 1.3, 4) 
	for (x, y, w, h) in faces: 
		cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
		face = im[y:y + h, x:x + w] 
		face_resize = cv2.resize(face, (width, height)) 
		cv2.imwrite('% s/% s.png' % (path, count), face_resize)
		sleep(1)
	count += 1
	cv2.imshow('OpenCV', im) 
	key = cv2.waitKey(10) 
	if key == 27: 
		break
