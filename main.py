import cv2 as cv
import numpy as np
import dlib
    
cap=cv.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(\"shape_predictor_68_face_landmarks.dat\")
    
while True:
    _,frame=cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
        print(face)
        x,y=face.left() ,face.top()
        x1,y1=face.right(), face.bottom()
        cv.rectangle(frame,(x,y),(x1,y1),(0,0,255),2)
        landmarks=predictor(gray,face)
        print(landmarks)
        x=landmarks.part(60).x
        y=landmarks.part(60).y
        cv.circle(frame,(x,y),3,(0,0,255),2)
        cv.imshow(\"Frame\",frame)
    
    key=cv.waitKey(0)
    if key == 27:
        break
cap.release()
cv.destroyAllWindows()