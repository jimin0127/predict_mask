import cv2
import cvlib as cv
import os

file = 1

img_dir = ['mask']

for dir in img_dir:
    for file_name in os.listdir(dir):
        img = cv2.imread(dir+'/'+file_name)
        print(dir+'/'+file_name)
        print(type(img))

        face, confidence = cv.detect_face(img)
        print(face)
        print(confidence)

        if face == None:
            print(dir+'/' + file_name)
            continue


        # loop through detected faces
        for idx, f in enumerate(face):
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            face_in_img = img[startY:endY, startX:endX, :]
            cv2.imwrite('img/' + '0/' + dir + str(file) + '.jpg', face_in_img)
            file += 1

        # display output
        cv2.imshow("captured frames", img)

        # release resources
        cv2.destroyAllWindows()
