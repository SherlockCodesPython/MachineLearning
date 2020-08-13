import os
from PIL import Image
import numpy as np
import cv2 as cv
import pickle

face_cascade = cv.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
recognoizer = cv.face.LBPHFaceRecognizer_create()
#gives the file path
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))

current_id = 0
label_ids = {}
y_labels = []
x_train =[]

BASE_DIR = (r"C:\Users\Anik Chatterjee\Untitled Folder")
image_dir = os.path.join(BASE_DIR,"images")

for root, dirs, files in os.walk(image_dir):
    for file in files:
       if file.endswith("png") or file.endswith("jpg"):
        path = os.path.join(root,file)
        label = os.path.basename(os.path.dirname(path)).replace(" ", "_").lower()
        #print(label,path)
        
        if not label in label_ids:
          label_ids[label] = current_id
        current_id +=1 
        id_ = label_ids[label]
        #print(label_ids)
        #y_label.append(label)
        #x_train.append(path)
        pil_image = Image.open(path).convert("L")
        image_array = np.array(pil_image,"uint8")
        print(image_array)
        faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5, minNeighbors=5)
        
        for (x,y,w,h) in faces:
           roi = image_array[y:y+w,x:x+h]
           x_train.append(roi) 
           y_labels.append(id_)
#print(y_label)
#print(x_train)

with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)
recognoizer.train(x_train,np.array(y_labels))
recognoizer.save("trainner.yml")
