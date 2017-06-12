import cv2
import numpy as np
from tkinter import *

class Slider:

    def __init__(self, master):

        frame = Frame(master)
        frame.pack()
        scale = Scale(frame, from=0, to=255, orient=HORIZONTAL, command=self.update)
        scale.grid(row=0)


    def update():
        
    

def main():

    img = cv2.imread("image/test1.jpg", 1)

    #
    # change road line to white
    #

    while True:

        img = changeColor(img)

        img_width = img.shape[1]
        img_height = img.shape[0]

        #
        # Turn color image into gray
        #

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #
        # BLUR image
        #

        #img = cv2.blur(img, (10, 10), 0)
        #cv2.imshow("Test", img)

        cv2.imwrite("image/new_image.jpg", img)

        k = cv2.waitKey()

        if k == 27:
            cv2.destroyAllWindows()

def slider():

    print("test")

def changeColor(img):


    # http://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(hls, lower, upper)

    # yellow mask
    lower = np.uint8([80, 120, 160])
    upper = np.uint8([160, 200, 255])
    yellow_mask = cv2.inRange(hls, lower, upper)

    #cv2.imshow('white_mask', white_mask)
    #cv2.imshow('yellow_mask', yellow_mask)

    # set masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    res = cv2.bitwise_and(img, img, mask=mask)

    #cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    return res


if __name__ == "__main__":
    main()

