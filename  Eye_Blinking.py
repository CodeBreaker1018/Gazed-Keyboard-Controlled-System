import cv2 as cv
import numpy as np
import dlib
from math import hypot

cap=cv.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1,p2):
    return int((p1.x+p2.x)/2) ,int((p1.y + p2.y)/2)
font = cv.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points,facial_landmarks):
    
    left_point = (facial_landmarks.part(36).x ,facial_landmarks.part(36).y)
    right_point = (facial_landmarks.part(39).x , facial_landmarks.part(39).y)
    center_top = midpoint(facial_landmarks.part(37) , facial_landmarks.part(38))
    center_bottom = midpoint(facial_landmarks.part(41) , facial_landmarks.part(40))
    hor_line = cv.line(frame , left_point ,right_point ,(0,255,0),2)
    ver_line = cv.line(frame , center_top ,center_bottom , (0,255,2),2)
    ### when we will get closer we will get high number vice versa
    ver_line_length = hypot((center_top[0]- center_bottom[0]) , (center_top[1]-center_bottom[1]))
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    #Eye Blinking Ratio
    ratio = (hor_line_length / ver_line_length)
while True:
    _,frame=cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
        
         
        x,y=face.left() ,face.top()
       # x1,y1=face.right(), face.bottom()
       # cv.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)
        landmarks = predictor(gray,face)
        left_point = (landmarks.part(36).x ,landmarks.part(36).y)
        right_point = (landmarks.part(39).x , landmarks.part(39).y)
        center_top = midpoint(landmarks.part(37) , landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41) , landmarks.part(40))
        hor_line = cv.line(frame , left_point ,right_point ,(0,255,0),2)
        ver_line = cv.line(frame , center_top ,center_bottom , (0,255,2),2)
        ### when we will get closer we will get high number vice versa
        ver_line_length = hypot((center_top[0]- center_bottom[0]) , (center_top[1]-center_bottom[1]))
        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        #Eye Blinking Ratio
        ratio = (hor_line_length / ver_line_length)
        
        if ratio > 5.7:
            cv.putText(frame,"BLINKING",(50,150) ,font , (255,0,0))
    
    cv.imshow("Frame",frame)

    key=cv.waitKey(1)
    if key == 27:
        break
cap.release()
cv.destroyAllWindows()