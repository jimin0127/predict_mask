import cv2
import cvlib as cv
import os



img_dir = ['mask', 'nomask']

for dir in img_dir:
    file = 1
    for file_name in os.listdir(dir):
        img = cv2.imread(dir+'/'+file_name)
        cv2.imwrite('img/'+dir + '/' + dir + str(file) + '.jpg', img)
        file += 1
cv2.destroyAllWindows()
