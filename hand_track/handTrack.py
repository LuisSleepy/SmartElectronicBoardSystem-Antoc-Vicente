import cv2 as cv
import numpy as np
import sys
sys.path.append("..")
from common import Sketcher

def nothing(x):
    pass


arrayX = []
arrayY = []
Kernal = np.ones((5, 5), np.uint8)

capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
_, frame = capture.read()
print(frame.shape)

import sys

try:
    fn = sys.argv[1]
except:
    fn = '../assets/fruits.jpg'

    img = cv.imread(cv.samples.findFile(fn))
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)
while (1):
    ret, frame = capture.read()
    frame = cv.flip(frame, +1)
    if not ret:
        break
    frame2 = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
    if cv.waitKey(1) == ord('q'):
        break

    lowerBoundary = np.array([0, 0, 81])
    upperBoundary = np.array([185, 255, 255])

    mask = cv.inRange(frame2, lowerBoundary, upperBoundary)
    cv.imshow('Mask', mask)

    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow('Res', res)

    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, Kernal)
    cv.imshow('Opening', opening)

    dilation = cv.dilate(opening, Kernal, iterations=5)
    cv.imshow('Dilation', dilation)

    contours, hierarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv.contourArea(contour) > 1500:
            cnt = contour

    M = cv.moments(cnt)
    centroidX = int(M['m10'] / M['m00'])
    centroidY = int(M['m01'] / M['m00'])
    cv.circle(frame, (centroidX, centroidY), 5, [50, 120, 255], -1)

    extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
    print(centroidX - extTop[0])

    if abs(centroidY - extTop[1]) > 200 and abs(centroidX - extTop[0]) < 150:
        arrayX.append(extTop[0])
        arrayY.append(extTop[1])

    img_mark = img.copy()
    mark = np.zeros(img.shape[:2], np.uint8)
    sketch = Sketcher('img', [img_mark, mark], lambda: ((255, 255, 255), 255))
    sketch.show()

    for i in range(len(arrayX)):
        cv.circle(capture, (arrayX[i], arrayY[1]), 4, (255, 155, 100), 5)

    cv.imshow('Resulting Image', frame)


capture.release()
cv.destroyAllWindows()