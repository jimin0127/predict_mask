import cv2 as cv
import numpy as np
import os
import cvlib
import threading

# 변경 사항 : 얼굴 인식을 cvlib.detect_face로 변경하여 얼굴 인식 정확도를 높임
# 검게 변하는 영역을 얼굴 영역으로 줄임

# 코드 수정
# 타이머 구현

def detect(img):
    face, confidence = cvlib.detect_face(img)

    if len(face) == 0:
        return []

    return face


def removeFaceAra(img):
    faces = detect(img)

    for x1, y1, x2, y2 in faces:
        cv.rectangle(img, (x1 - 10, y1), (x2 + 10, y2), (0, 0, 0), -1)

    return img


def make_mask_image(img_bgr):
    img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

    low = (0, 30, 0)
    high = (15, 255, 255)

    img_mask = cv.inRange(img_hsv, low, high)
    return img_mask


def distanceBetweenTwoPoints(start, end):
    x1, y1 = start
    x2, y2 = end

    return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))


def calculateAngle(A, B):
    # 거리 계산(벡터 끼리의 연산)
    A_norm = np.linalg.norm(A)
    B_norm = np.linalg.norm(B)
    C = np.dot(A, B)

    # 각도 계산
    angle = np.arccos(C / (A_norm * B_norm)) * 180 / np.pi
    return angle


def findMaxArea(contours):
    max_contour = None
    max_area = -1

    for contour in contours:
        area = cv.contourArea(contour)

        x, y, w, h = cv.boundingRect(contour)

        if (w * h) * 0.4 > area:
            continue

        if w > h:
            continue

        if area > max_area:
            max_area = area
            max_contour = contour

    if max_area < 10000:
        max_area = -1

    return max_area, max_contour


def getFingerPosition(max_contour, img_result, debug):
    points1 = []

    # STEP 6-1
    M = cv.moments(max_contour)

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    max_contour = cv.approxPolyDP(max_contour, 0.02 * cv.arcLength(max_contour, True), True)
    hull = cv.convexHull(max_contour)

    for point in hull:
        if cy > point[0][1]:
            points1.append(tuple(point[0]))

    if debug:
        cv.drawContours(img_result, [hull], 0, (0, 255, 0), 2)
        for point in points1:
            cv.circle(img_result, tuple(point), 15, [0, 0, 0], -1)

    # STEP 6-2
    hull = cv.convexHull(max_contour, returnPoints=False)
    defects = cv.convexityDefects(max_contour, hull)

    if defects is None:
        return -1, None

    points2 = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])

        angle = calculateAngle(np.array(start) - np.array(far), np.array(end) - np.array(far))

        if angle < 90:
            if start[1] < cy:
                points2.append(start)

            if end[1] < cy:
                points2.append(end)

    if debug:
        cv.drawContours(img_result, [max_contour], 0, (255, 0, 255), 2)
        for point in points2:
            cv.circle(img_result, tuple(point), 20, [0, 255, 0], 5)

    # STEP 6-3
    points = points1 + points2
    points = list(set(points))

    # STEP 6-4
    new_points = []
    for p0 in points:

        i = -1
        for index, c0 in enumerate(max_contour):
            c0 = tuple(c0[0])

            if p0 == c0 or distanceBetweenTwoPoints(p0, c0) < 20:
                i = index
                break

        if i >= 0:
            pre = i - 1
            if pre < 0:
                pre = max_contour[len(max_contour) - 1][0]
            else:
                pre = max_contour[i - 1][0]

            next = i + 1
            if next > len(max_contour) - 1:
                next = max_contour[0][0]
            else:
                next = max_contour[i + 1][0]

            if isinstance(pre, np.ndarray):
                pre = tuple(pre.tolist())
            if isinstance(next, np.ndarray):
                next = tuple(next.tolist())

            angle = calculateAngle(np.array(pre) - np.array(p0), np.array(next) - np.array(p0))

            if angle < 90:
                new_points.append(p0)

    return 1, new_points

# time_count = 3

# def timer_start():
#     global time_count
#     print(time_count)
#     time_count -= 1
#
#     timer = threading.Timer(1, print(time_count))
#     timer.start()
#
#     if time_count == 0:
#         timer.cancel()
#         ret, v_image = cap.read()
#         cv.imwrite('v.jpg', v_image)


def process(img, debug):
    img_result = img.copy()

    # STEP 1
    img_bgr = removeFaceAra(img)

    # STEP 2
    img_binary = make_mask_image(img_bgr)

    # STEP 3
    # 타원형 터널
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    # 작은 구멍이나 검은 검을 없앤다
    img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)
    cv.imshow("Binary", img_binary)

    # STEP 4
    # contours : 동일한 색 또는 동일한 색상 강도를 가진 가장 자리를 연결한 선
    # RETR_EXTERNAL : 이미지의 가장 바깥쪽의 contours만 추출
    # CHAIN_APPROX_SIMPLE : 모든 점을 저장하지 않고 끝점만 저장
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if debug:
        for cnt in contours:
            # 가장 자리를 그림
            cv.drawContours(img_result, [cnt], 0, (255, 0, 0), 3)

            # STEP 5
    max_area, max_contour = findMaxArea(contours)

    if max_area == -1:
        return img_result

    if debug:
        cv.drawContours(img_result, [max_contour], 0, (0, 0, 255), 3)

        # STEP 6
    ret, points = getFingerPosition(max_contour, img_result, debug)

    # STEP 7
    if ret > 0 and len(points) > 0:
        if len(points) == 2:
            #timer_start()
            # 타이머
            print('2')
        for point in points:
            cv.circle(img_result, point, 20, [255, 0, 255], 5)

    return img_result


# cap = cv.VideoCapture('test.avi')

cap = cv.VideoCapture(0)

while True:

    ret, img_bgr = cap.read()

    if ret == False:
        break


    img_result = process(img_bgr, debug=False)

    key = cv.waitKey(1)
    if key == 27:
        break


    cv.imshow("Result", img_result)

cap.release()
cv.destroyAllWindows()
