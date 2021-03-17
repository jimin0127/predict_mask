# 카메라에서 얼굴 영역 검출 후, 마스크 착용 여부 판별하기

import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image

model = load_model('model.h5')
model.summary()

# 웹캠 열기
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():

    # 웹캠에서 프레임 읽어오기
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # 얼굴 감지하기
    face, confidence = cv.detect_face(frame)

    # 감지된 얼굴만큼 반복하기
    for idx, f in enumerate(face):

        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:

            face_region1 = frame[startY:endY, startX:endX]

            face_region2 = cv2.resize(face_region1, (224, 224), interpolation = cv2.INTER_AREA)

            x = img_to_array(face_region2)
            x = np.expand_dims(x, axis=0)
            x=preprocess_input(x)

            prediction = model.predict(x)

            # 마스크 미착용으로 판별된다면
            if prediction < 0.5:
                cv2.rectangle((frame), (startX, startY), (endX, endY), (0, 0, 255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "No Mask(({:.2f}%)".format((1 - prediction[0][0])*100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

            # 마스크 착용으로 판별된다면
            else:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "Mask ({:.2f}%)".format(prediction[0][0]*100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # display output
        cv2.imshow("mask nomask classify", frame)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#release resources
webcam.release()
cv2.destroyAllWindows()

