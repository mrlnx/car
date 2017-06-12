import cv2
from tkinter import *
import os
from PIL import Image, ImageTk
import numpy as np

class HSLSlider:

    def __init__(self, root):

        self.h = 0
        self.s = 0
        self.k = 0
        
        self.img = cv2.imread("image/test.jpg", 1)
        self.img = cv2.resize(self.img, (1280,800))
        
        #cv2.imshow("show im", self.img)

##        capture = cv2.VideoCapture("video/project_video1.mp4")
##
##        print(capture)
##
##        while(capture.isOpened()):
##
##            ret, self.img = capture.read()
##
##            #cv2.imshow("show im", self.img)
##            
##            if cv2.waitKey(1) & 0xFF == ord("q"):
##                break

        image = Image.fromarray(self.img)
        image = ImageTk.PhotoImage(image)

        # top image frame
        self.top_frame = Frame(root)
        self.top_frame.pack()

        self.top_panel = Label(self.top_frame, image=image)
        self.top_panel.image = image
        self.top_panel.pack()

        bottom_frame = Frame(root)
        bottom_frame.pack(side=LEFT)

        slider_label = Label(bottom_frame, text="Yellow slider")
        slider_label.grid(row=0)

        h_slider = Scale(bottom_frame, length=255, from_=0, to=255, orient=HORIZONTAL, command=lambda value, name="h": self.update(name, value))
        h_slider.grid(row=1)

        s_slider = Scale(bottom_frame, length=255, from_=0, to=255, orient=HORIZONTAL, command=lambda value, name="s": self.update(name, value))
        s_slider.grid(row=2)

        l_slider = Scale(bottom_frame, length=255, from_=0, to=255, orient=HORIZONTAL, command=lambda value, name="l": self.update(name, value))
        l_slider.grid(row=3)

        # show
        #cv2.imshow("Showimg", img)

        # always
        k = cv2.waitKey()

        if k == 27:
            cv2.destroyAllWindows()

       # capture.release()
        #cv2.destroyAllWindows()

    def update(self, name, value):

        if name == "h":
            self.h = value
        elif name == "s":
            self.s = value
        elif name == "l":
            self.l = value

        self.hsl = (self.h, self.s, self.l)
        self.change(self.hsl)

        
    def change(self, hsl):

        self.h, self.s, self.l = hsl

        print(hsl)

        #upper = np.uint8([160, 200, 255])
        #lower = np.uint8([self.h, self.s, self.l])

        hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)

        lower = np.uint8([0, 200, 50])
        upper = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(hls, lower, upper)

        # yellow mask

        lower = np.uint8([self.h, self.s, self.l])
        #lower = np.uint8([80, 130, 160])
        # rgb(70, 80, 73)
        upper = np.uint8([160, 200, 255])
        
        
        yellow_mask = cv2.inRange(hls, lower, upper)

        #cv2.imshow('white_mask', white_mask)
        #cv2.imshow('yellow_mask', yellow_mask)

        # set masks
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        res = cv2.bitwise_and(self.img, self.img, mask=mask)

        image = Image.fromarray(res)
        image = ImageTk.PhotoImage(image)

        self.top_panel.configure(image=image)
        self.top_panel.image = image
        

if __name__ == "__main__":

    root = Tk()
    root.wm_title("HSL Slider")

    hslSlider = HSLSlider(root)
    
    root.mainloop()
