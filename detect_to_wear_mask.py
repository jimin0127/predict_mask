# 카메라에서 얼굴 영역 검출 후, 마스크 착용 여부 판별하기

import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image


class detect_to_wear_mask():


    def __init__(self, frame):
        self.model = load_model('model.h5')
        self.model.summary()
        self.frame = frame

    def test(self):
        a = 0
        # 얼굴 감지하기
        face, confidence = cv.detect_face(self.frame)

        # 감지된 얼굴만큼 반복하기
        for idx, f in enumerate(face):

            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            if 0 <= startX <= self.frame.shape[1] and 0 <= endX <= self.frame.shape[1] and 0 <= startY <= self.frame.shape[0] and 0 <= endY <= self.frame.shape[0]:

                face_region = self.frame[startY:endY, startX:endX]

                face_region1 = cv2.resize(face_region, (224, 224), interpolation = cv2.INTER_AREA)

                x = img_to_array(face_region1)
                x = np.expand_dims(x, axis=0)
                x=preprocess_input(x)

                prediction = self.model.predict(x)

                # 마스크 미착용으로 판별된다면
                if prediction < 0.5:
                    cv2.rectangle(self.frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    text = "No Mask(({:.2f}%)".format((1 - prediction[0][0])*100)
                    cv2.putText(self.frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 마스크 착용으로 판별된다면(1)
                else:
                    cv2.rectangle(self.frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    text = "Mask ({:.2f}%)".format(prediction[0][0]*100)
                    cv2.putText(self.frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    a += 1


        if a == 6:
            return 1



if __name__ == '__main__':
    d = detect_to_wear_mask()
    print(d.test())