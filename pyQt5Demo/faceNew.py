#使用opencv+python实现最简单摄像头实时人脸识别（适合入门）
import cv2
cap = cv2.VideoCapture(0)
face = cv2.CascadeClassifier("D:\Files\PythonFile\Python\data\haarcascades\haarcascade_frontalface_default.xml")
while(1):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    faces = face.detectMultiScale(gray,1.1,3,0,(200,200))
    for (x,y,w,h) in faces:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        gray_roi = gray[y:y+h,x:x+h]
        cv2.imwrite('face.jpg',gray_roi)
        print('Image saved!!')
    cv2.imshow('摄像头',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

