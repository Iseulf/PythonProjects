from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget, QApplication
from PyQt5.QtGui import QIcon, QPixmap
import cv2
from keras.models import load_model
import numpy as np
import ChineseText
import sys
from time import sleep
import os


class WindowClass(QWidget):
    def __init__(self, parent=None):
        super(WindowClass, self).__init__(parent)
        self.cap = cv2.VideoCapture(0)
        self.ret = None
        self.img = None
        self.number = 1
        self.classifier = cv2.CascadeClassifier(
            "D:\Files\PythonFile\pyQt5Demo\data\haarcascades\haarcascade_frontalface_default.xml")

        self.btn_1 = QPushButton("开始")
        self.btn_3 = QPushButton("捕捉")
        self.btn_4 = QPushButton("退出")
        self.btn_2=QPushButton("特效")

        # self.btn_1.setCheckable(True)  # 设置已经被点击
        # self.btn_1.toggle()  # 切换按钮状态
        self.btn_1.clicked.connect(lambda: self.wichBtn(self.btn_1))

       # self.btn_2.setDefault(True)
        self.btn_2.clicked.connect(lambda: self.wichBtn(self.btn_2))

       # self.btn_3.setDefault(True)  # 设置该按钮式默认状态的
        self.btn_3.clicked.connect(lambda: self.wichBtn(self.btn_3))

        self.btn_4.clicked.connect(lambda: self.wichBtn(self.btn_4))

        self.resize(400, 300)
        layout = QVBoxLayout()
        layout.addWidget(self.btn_1)
        layout.addWidget(self.btn_2)
        layout.addWidget(self.btn_3)
        layout.addWidget(self.btn_4)
        # layout.addWidget(self.btn_2)

        self.setLayout(layout)

    def btnState(self):

        # OpenCV人脸识别分类器
        # cap=cv2.VideoCapture(0)
        while(1):
            self.ret, self.img = self.cap.read()
            gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
            gender_classifier = load_model(
    "D:\Files\PythonFile\pyQt5Demo\\faceai-master\\faceai\classifier\gender_models\simple_CNN.81-0.96.hdf5")
            gender_labels = {0: '女', 1: '男'}
            # classifier = cv2.CascadeClassifier(
            #     "D:\\Files\PythonFile\\pyQt5Demo\\data\haarcascades\\haarcascade_frontalface_default.xml"
            # )
            color = (0, 100, 0)  # 定义绘制颜色
            # 调用识别人脸
            faceRects = self.classifier.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faceRects):  # 大于0则检测到人脸
                for faceRect in faceRects:  # 单独框出每一张人脸
                    x, y, w, h = faceRect
                    face = self.img[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
                    face = cv2.resize(face, (48, 48))
                    face = np.expand_dims(face, 0)
                    face = face / 255.0 
                    gender_label_arg = np.argmax(gender_classifier.predict(face))
                    gender = gender_labels[gender_label_arg]
                    cv2.rectangle(self.img, (x, y), (x + h, y + w), color, 2)
                    self.img = ChineseText.cv2ImgAddText(self.img, gender, x + h, y, color, 30)
                    # 框出人脸
                    cv2.rectangle(self.img, (x, y), (x + h, y + w), color, 1)
                    # 左眼
                    cv2.circle(self.img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                               color, 1)
                    # 右眼
                    cv2.circle(self.img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                               color, 1)
                    # 嘴巴
                    cv2.rectangle(self.img, (x + 3 * w // 8, y + 3 * h // 4),
                                  (x + 5 * w // 8, y + 7 * h // 8), color, 1)

            cv2.namedWindow("enhanced", flags=0)
            cv2.resizeWindow("enhanced", 1000, 960)
            cv2.imshow("enhanced", self.img)  # 显示图像
            c = cv2.waitKey(10)
    def btnState2(self):
        while(1):
            imgCompose = cv2.imread("D:\Files\PythonFile\\figures\maozi.png")
           # cap = cv2.VideoCapture(0)
            ret,img=self.cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰色
            color = (0, 255, 0)  # 定义绘制颜色
            # 调用识别人脸
            faceRects = self.classifier.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faceRects):  # 大于0则检测到人脸
                for faceRect in faceRects:  
                    x, y, w, h = faceRect
                    sp = imgCompose.shape
                    imgComposeSizeH = int(sp[0]/sp[1]*w)
                    if imgComposeSizeH>(y-20):
                        imgComposeSizeH=(y-20)
                    imgComposeSize = cv2.resize(imgCompose,(w, imgComposeSizeH), interpolation=cv2.INTER_NEAREST)
                    top = (y-imgComposeSizeH-20)
                    if top<=0:
                        top=0
                    rows, cols, channels = imgComposeSize.shape
                    roi = img[top:top+rows,x:x+cols]

                    # Now create a mask of logo and create its inverse mask also
                    img2gray = cv2.cvtColor(imgComposeSize, cv2.COLOR_RGB2GRAY)
                    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY) 
                    mask_inv = cv2.bitwise_not(mask)

                    # Now black-out the area of logo in ROI
                    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                    # Take only region of logo from logo image.
                    img2_fg = cv2.bitwise_and(imgComposeSize, imgComposeSize, mask=mask)

                    # Put logo in ROI and modify the main image
                    dst = cv2.add(img1_bg, img2_fg)
                    img[top:top+rows, x:x+cols] = dst
            cv2.namedWindow("enhanced", flags=0)
            cv2.resizeWindow("enhanced", 1000, 960)
            cv2.imshow("enhanced", img)  # 显示图像
            c = cv2.waitKey(10)
    def wichBtn(self, btn):
        if btn == self.btn_1:
            self.btnState()
        elif btn == self.btn_4:
            self.cap.release()
            self.close()
        elif btn==self.btn_2:
            self.btnState2()
        else:
            imgpath = "D:\Files\PythonFile\\figures" "\\" + \
                str(self.number) + ".jpg"
            cv2.imwrite(imgpath, self.img)
            self.number = self.number+1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WindowClass()
    win.show()
    sys.exit(app.exec_())
