import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from time import sleep
import csv

hlines = True

def main(img):
    
    original = img

    # convert yellow colors to white
    img = cvtYellow(img)

    # compute region of intrest
    vertex = computeROI(img)

    if hlines:

        # regio of intrehsl hsv machine learning st polylines
        cv2.polylines(original, vertex, 1, (0,255,255), 2)

        # vertical line
        cv2.line(original, (math.ceil(original.shape[1] * 0.5), 0), (math.ceil(original.shape[1] * 0.5), original.shape[0]), (0, 255, 0), 3)

        # horizontal line
        cv2.line(original, (0, math.ceil(original.shape[0] * 0.5)), (original.shape[1], math.ceil(original.shape[0] * 0.5)), (0, 255, 0), 3)

        angles = [[-25, 60, 660],
                  [-55, 60, 660],
                  [-125, 1220, 660],
                  [-155, 1220, 660]] 

        max_width = img.shape[1]
        max_height = img.shape[0] - 60

        for angle in angles:

            cangle = angle[0]
            length = 700

            p1_x = angle[1]
            p1_y = angle[2]
            
            theta = cangle * math.pi / 180

            p2_x = round(p1_x + length * math.cos(cangle * math.pi / 180))
            p2_y = round(p1_y + length * math.sin(cangle * math.pi / 180))

            cv2.line(original, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), (0, 0, 255), 3)
            cv2.circle(original, (p1_x, p1_y), 4, (0,10,0), 4)
            cv2.circle(original, (p2_x, p2_y), 4, (255,0,), 4)


##        angles = [[25], [75], [105], [155]] 
##
##        max_width = img.shape[1]
##        max_height = img.shape[0] - 60
##        bottom_width = 580
##
##        angle = 25
##        length = 1280
##        p1_x = (max_width / 2) - bottom_width
##        p1_y = max_height
##
##        theta = angle * math.pi / 180
##
##        p2_x = round(p1_x + length * math.cos(angle * math.pi / 180))
##        p2_y = round(p2_y + length * math.cos(angle * math.pi / 180))
##
##        cv2.line(original, (line[0], line[1]), (line[2], line[3]), (0, 0, 0), 3)


    # 
    img = getRegion(img, vertex)

    #cv2.imshow("Regio", img)

    # convert bgr2gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("cvtColor", img)

    # blur
    size = 7
    img = cv2.GaussianBlur(img, (size, size), 0)

    # canny edge detection
    low_thres = 100
    high_thres = 300

    img = cv2.Canny(img, low_thres, high_thres)

    #cv2.imshow("IMG", img)

    #merge original with lines
    edges = np.dstack((img, img, img))
    test = cv2.addWeighted(edges, 2, original, 1, 0)

    #cv2.imshow("Image", test)

    drawLines(img, original)

    cv2.imshow("Original", original)

def cvtYellow(img):
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # white mask
    lower = np.uint8([0, 200, 50])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(hls, lower, upper)

    # yellow mask
    lower = np.uint8([80, 130, 160])
    upper = np.uint8([160, 200, 255])
        
    yellow_mask = cv2.inRange(hls, lower, upper)
        
    # set masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    result = cv2.bitwise_and(img, img, mask=mask)

    return result

def computeROI(img):

        top_width = 180
        bottom_width = 580
    
        top_width = (top_width / 2) + 30
        #bottom_width = (bottom_width / 2)

        max_width = img.shape[1]
        max_height = img.shape[0] - 60

        width_delta = int(max_width/16)

        bottom_left = ((max_width / 2) - bottom_width, max_height)
        bottom_right = ((max_width / 2) + bottom_width, max_height)

        top_left = ((max_width / 2) - bottom_width, max_height - 230)
        top_right = ((max_width / 2) + bottom_width, max_height - 230)

        #top_left = (max_width / 2 - width_delta, max_height / 2 + top_width)
        #top_right = (max_width / 2 + width_delta, max_height / 2 + top_width)

        #print(bottom_left)
        #print(bottom_right)

        array = np.array([[bottom_left, bottom_right, top_right, top_left]], np.int32)
        
        return array

def getRegion(img, vertex):

    mask = np.zeros_like(img)
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertex, ignore_mask_color)


    return cv2.bitwise_and(img, mask)


def houghLines(img):

    rho = 0.8
    theta = np.pi / 180
    thres = 25
    min_line_length = 50
    max_line_gap = 200

    hough = cv2.HoughLinesP(img, rho, theta, thres, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

    if(hough is None or len(hough) > 500):
        return []
    else:
        return hough
    
def evaluateLine(line, img, original):
    
    lines = houghLines(img)
    lines_original = np.zeros(img.shape, dtype=np.uint8)
    line_width = 2

    width = img.shape[0]
    height = img.shape[1]
    
    lines_list = []

    color = (0, 0, 255)
    
    try:
        
        for x1, y1, x2, y2 in line:

            y1 = -(y1 - height) 
            y2 = -(y2 - height) 

            #print(str(x1) + " => " + str(x2))
            #print(str(y1) + " => " + str(y2))
            
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2

            center = (x, y)
            angle = math.atan2((y2 - y1), (x2 - x1)) * 180 / math.pi



            if math.fabs(angle) <= 10:
                continue
            
            if x == 0:
                qqdx = 1

            dx = (x2 - x1)
            dy = (y2 - y1)

            k = dy / dx
            b = y1 - k * x1

            if x2 == x1:
                continue

            if angle < 0.0:
                angle += 180

            #print(angle)

            #current_slope = (y2 - y1) / (x2 - x1)
            #intercept = y1 - current_slope * x1

            #x_min = x1
            #y_min = y1
            #x_max = x2
            #y_max = y2

            #print("angle = " + str(angle))
    
            lines_list.append((line, angle, center, x, b))
            
    except TypeError:
        print("Type Error")

    return lines_list

def drawLines(img, original):

    lines = houghLines(img)
    lines_original = np.zeros(img.shape, dtype=np.uint8)
    line_width = 2

    color = (0, 0, 255)

    # sortlines

    leftLines = []
    rightLines = []

    for line in lines:

        gradient, intercept = np.polyfit((line[0][0], line[0][1]), (line[0][2], line[0][3]), 1)

        print(gradient)

    for line in lines:

        #evaluatedLine = evaluateLine(line, img, original)
        
        for eLItem in evaluateLine(line, img, original):
            
            (x, y) = eLItem[2]
            angle = eLItem[1]

            #print("angle = " + str(angle))

            # eliminate lines
            if x < img.shape[1] * 0.5 and angle >= 25 and angle <= 55:
                leftLines.append(eLItem)
            #else:
            #    print(angle)

            
            if x > img.shape[1] * 0.5and angle >= 125 and angle <= 155:
                rightLines.append(eLItem)
            #else:
                #print(angle)

    for ll in leftLines:
        line = ll[0][0]
        center = ll[2]
        
        cv2.line(original, (line[0], line[1]), (line[2], line[3]), (255, 255, 255), 1)


    for rl in rightLines:
        line = rl[0][0]
        center = ll[2]
        
        cv2.line(original, (line[0], line[1]), (line[2], line[3]), (255, 255, 255), 1)


##    processLine(rightLines, original, True)
##    processLine(leftLines, original, False)

      
##    for ll in leftLines:
##        for rl in rightLines:
##
##            ll_x, ll_y = ll[2]            
##            cv2.circle(original, (math.ceil(ll_x), math.ceil(ll_y)), 4, (255,0,0), 4)
##
##            rl_x, rl_y = rl[2]
##            cv2.circle(original, (math.ceil(rl_x), math.ceil(rl_y)), 4, (0,0,255), 2)


def processLine(lane, img, right):

    #print(right)

    leftLine = []
    rightLine = []

    for i in range(0, len(lane)):

        if right:
            rightLine.append(lane[i][2])
            
        else:
            leftLine.append(lane[i][2])

    
    if right == True:
        rightLine = sorted(rightLine, key=lambda x: x[1])
        #cv2.line(img, (int(rightLine[0][0]), int(rightLine[0][1])), (int(rightLine[len(rightLine)-1][0]), int(rightLine[len(rightLine)-1][1])), (0, 0, 0), 3)
    else:
        leftLine = sorted(leftLine, key=lambda x: x[1])
        #cv2.line(img, (int(leftLine[0][0]), int(leftLine[0][1])), (int(leftLine[len(leftLine)-1][0]), int(leftLine[len(leftLine)-1][1])), (0, 0, 0), 3)
 
if __name__ == "__main__":

    content_type = "video"

    if content_type == "image":
        img = cv2.imread("image/test1.jpg", 1)
        img = cv2.resize(img, (1280,720))
        
        main(img)

        k = cv2.waitKey()

        if k == 27:
            cv2.destroyAllWindows()

    elif content_type == "video":
        
        capture = cv2.VideoCapture("video/project_video.mp4")

        while(capture.isOpened()):

            ret, img = capture.read()
            #img = cv2.resize(img, (1280, 720))

            #cv2.imshow("Video", img)

            main(img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        capture.release()
        cv2.destroyAllWindows()
        

    
