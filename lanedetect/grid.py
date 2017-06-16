import cv2
import numpy as np
from matplotlib import pyplot as plt

img0 = cv2.imread("image/test1.jpg")
img0 = cv2.resize(img0, (320, 180))

img1 = np.zeros(img0.shape, np.uint8)
img2 = np.zeros(img0.shape, np.uint8)

res = cv2.merge((img1, img2))

print(res)

cv2.imshow("Original1", res)

k = cv2.waitKey()

if k == 27:
    cv2.destroyAllWindow()
