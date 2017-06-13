import cv2
import math

img = cv2.imread("image/test.jpg", 1)
img = cv2.resize(img, (1280,720))

h = img.shape[0]
w = img.shape[1]

print(h)
print(w)

length = 650
angle = -105
px1 = w - 100
py1 = 660


t = angle * math.pi / 180

print(t)

px2 = round(px1 + length * math.cos(t))
py2 = round(py1 + length * math.sin(t))

cv2.line(img, (px1, py1), (px2, py2), (0, 0, 255), 3)

cv2.circle(img, (px1, py1), 4, (0,10,0), 4)
cv2.circle(img, (px2, py2), 4, (255,0,), 4)

angle = -155

t = angle * math.pi / 180

print(t)

px2 = round(px1 + length * math.cos(t))
py2 = round(py1 + length * math.sin(t))

cv2.line(img, (px1, py1), (px2, py2), (0, 0, 255), 3)

cv2.circle(img, (px1, py1), 4, (0,10,0), 4)
cv2.circle(img, (px2, py2), 4, (255,0,), 4)

cv2.imshow("Image", img)

k = cv2.waitKey()

if k == 27:
    cv2.destroyAllWindows()
