#This is for opening web cam and detecting your face and emotion

import cv2
from deepface import DeepFace


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("cannot open")

while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, actions=[ 'emotion' ])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4) #dectecting face

    #for drawing rectangle on detected face
    for(x, y, u, v) in faces:
        cv2.rectangle(frame, (x, y), (x+u, y+v), (0,255,0), 2)

    #for putting text on img
    cv2.putText(frame, result['dominant_emotion'], (50, 50), font, 3, (0,0,255), 2, cv2.LINE_4)
    
    cv2.imshow('original video', frame)

    #for closing cam press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()