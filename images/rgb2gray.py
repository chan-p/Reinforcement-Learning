import cv2
import numpy as np

im = cv2.imread("red.png")
# RGB空間からグレースケール空間
# print(im)

gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# print(gray)
gray.resize(128, 128)
# print(len(gray[0]))
cv2.imwrite("test.png",gray)
cv2.imshow("img", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
