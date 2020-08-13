import cv2 as cv
import numpy as np
import pickle 
#used to find the frontal face.make sure to convert color to gray
face_cascade = cv.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
recognoizer = cv.face.LBPHFaceRecognizer_create()

recognoizer.read("trainner.yml")

labels ={"person_name": 1}

with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
    

#face_eye_cascade = cv.CascadeClassifier("cascades/data/haarcascade_eye_tree_eyeglasses")

#to run the video module
vid_img = cv.VideoCapture(0)
#Infinite loop for video capture
while(True):
    #reading video image from camera
    ret,frame = vid_img.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    all_faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
    #all_eye = face_eye_cascade.detectMultiScale(all_faces)
    #roi_gray = gray[100:300,100:300]
    #cv.rectangle(frame,(100,100),(300,300),(0,255,0),0)
    for(x,y,w,h) in all_faces:
        #print(x,y,w,h)
        #[Region of interest :- ycoordinate start:- [coordinate  1 + height] ,y coordinate end[coordinate 2 +height]
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = gray[y:y+h,x:x+w]
        id_,conf = recognoizer.predict(roi_gray)
        if conf>= 25 and conf <= 85:
            #print(id_)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame,labels[id_],(0,50),font,1,(0,0,255),3, cv.LINE_AA)
            print(labels[id_])
        img_item = "my-image.png"
        cv.imwrite(img_item,roi_gray)
        #Draw rect
        color = (0,255,0)
        stroke = 5
        #end coordinates x
        width = x+w
        #End coordinates y
        height = y+h
        cv.rectangle(frame, (x,y), (width, height),color, stroke)
   #showing the image
    cv.imshow('frame',frame)
       
    
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
if (labels[id_]) == 'anik':
    print("Unlocked")
else:
    print('Locked Please try again')
vid_img.release()
cv.destroyAllWindows()
