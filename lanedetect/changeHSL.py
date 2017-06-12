import cv2
import numpy as np
from tkinter import *

class HSLSlider:

    def __init__(self, master):

        
        
        frame = Frame(master)
        
        
        h_slider = Scale(frame, from_=0, to=255, orient=HORIZONTAL, command=lambda value, name="h": self.update(name, value))
        h_slider.grid(row=0)

        s_slider = Scale(frame, from_=0, to=255, orient=HORIZONTAL, command=lambda value, name="s": self.update(name, value))
        s_slider.grid(row=1)

        l_slider = Scale(frame, from_=0, to=255, orient=HORIZONTAL, command=lambda value, name="l": self.update(name, value))
        l_slider.grid(row=2)

        hsl = HSL(master, (0,0,0))

        frame.pack()


    def update(self, name, value):

        if name == "h":
            self.h = value
        elif name == "s":
            self.s = value
        elif name == "l":
            self.l = value

        print("h = " + str(self.h) + " s = " + str(self.s) + " l = " + str(self.l))

        self.hsl = (self.h, self.s, self.l)


class HSL:

    def __init__(self, master, hsl):
        
        img = cv2.imread("image/test1.jpg", 1)

        #
        # change road line to white
        #

        while True:

            img = self.changeColor(img)

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

    def changeColor(self, img):

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

    #hsl = HSL((0,0,0))

    root = Tk()
    root.wm_title("HSL sliders")
    slider = HSLSlider(root)
    root.geometry("200x150+0+0")

    
    
    root.mainloop()

    
