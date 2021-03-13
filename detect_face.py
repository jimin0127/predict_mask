import cv2
import cvlib as cv

# open webcam (웹캠 열기)
webcam = cv2.VideoCapture(0)
#capture = cv2.VideoCapture('sample.mp4')
# 너비 : 640
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# 높이 : 480
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

sample_num = 0
captured_num = 0

# loop through frames
while cv2.waitKey(33) < 0:

    # read frame from webcam
    status, frame = webcam.read()
    sample_num = sample_num + 1

    if not status:
        break

    # 이미지 내 얼굴 검출
    face, confidence = cv.detect_face(frame)

    print(face)
    print(confidence)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # loop through detected faces
    for idx, f in enumerate(face):

        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        if sample_num % 8 == 0:
            captured_num = captured_num + 1
            face_in_img = frame[startY:endY, startX:endX, :]
            cv2.imwrite('mask/face' + str(captured_num) + '.jpg', face_in_img)  # 마스크 미착용 데이터 수집시 주석처리
            # cv2.imwrite('./nomask/face'+str(captured_num)+'.jpg', face_in_img) # 마스크 미착용 데이터 수집시 주석해제

    # display output
    cv2.imshow("captured frames", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
