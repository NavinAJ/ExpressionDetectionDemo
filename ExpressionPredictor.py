import cv2
import pickle
import numpy as np
import os
from keras.preprocessing.image import img_to_array



class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

def PredictEmotion(app,filename,_fileDirectory):
    face_classifier = cv2.CascadeClassifier(os.path.join(app.root_path,'haarcascade_frontalface_default.xml'))
    ExpressionPickleModel = pickle.load(open(os.path.join(app.root_path,'Expression.pkl'),'rb'))
    
    frame = cv2.imread(_fileDirectory)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            # make a prediction on the ROI, then lookup the class
            preds = ExpressionPickleModel.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            imageFrame = cv2.imwrite(_fileDirectory, frame)
            data = {"html":"predict.html","image_name":filename,"message":label}
            return data
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            data = {"html":"predict.html","image_name":filename,"message":"No face detected"}
            return data
    data = {"html":"predict.html","image_name":filename,"message":"No face detected"}
    return data
    
