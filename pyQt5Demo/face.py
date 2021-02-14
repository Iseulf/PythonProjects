import cv2

cap = cv2.VideoCapture(0)

# OpenCV人脸识别分类器
while(1):
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    classifier = cv2.CascadeClassifier(
        "D:\Files\PythonFile\pyQt5Demo\data\haarcascades\haarcascade_frontalface_default.xml"
    )
    color = (0, 0, 0)  # 定义绘制颜色
    # 调用识别人脸
    faceRects = classifier.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects):  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            # 框出人脸
            cv2.rectangle(img, (x, y), (x + h, y + w), color, 4)
            # 左眼
            cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                    color,5)
            #右眼
            cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                    color,5)
            #嘴巴
            cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4),
                        (x + 5 * w // 8, y + 7 * h // 8), color,4)
           # img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
           # gray_roi = gray[y:y+h,x:x+h]
            # cv2.imwrite('face.jpg',img)
            print('Image saved!!')
    cv2.imwrite('face.jpg',img)
    cv2.namedWindow("enhanced",flags=0)
    cv2.resizeWindow("enhanced", 640, 480)
    cv2.imshow("enhanced", img)  # 显示图像
    c = cv2.waitKey(10)

cv2.waitKey(0)
cv2.destroyAllWindows()