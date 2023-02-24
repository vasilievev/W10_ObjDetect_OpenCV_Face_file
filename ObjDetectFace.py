import cv2
import cvlib as cv
import numpy as np

cap=cv2.VideoCapture("YOU_FILE.mp4")

while True:
    ret,frame=cap.read()
    if not ret:
        break
        
    frame=cv2.resize(frame,(1280,1024))
    faces,confidences=cv.detect_face(frame)
    
    for face,conf in zip(faces,confidences):
        x,y=face[0],face[1]
        x1,y1=face[2],face[3]
        crop=frame[y:y1,x:x1]
        (label,confidence)=cv.detect_gender(crop)
        idx=np.argmax(confidence)
        label=label[idx]
        print(label)
        cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)
        cv2.putText(frame,str(label),(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),2)
    cv2.imshow("Frame",frame)
    
    if cv2.waitKey(1)&0xFF==27:
        break
        
cap.release()
cv2.destroyAllWindows()
