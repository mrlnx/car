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
        cv2.polylines(original, vertex, 1, (0,255,0), 2)

        # vertical line
        cv2.line(original, (math.ceil(original.shape[1] * 0.5), 0), (math.ceil(original.shape[1] * 0.5), original.shape[0]), (0, 255, 0), 3)

        # horizontal line
        cv2.line(original, (0, math.ceil(original.shape[0] * 0.5)), (original.shape[1], math.ceil(original.shape[0] * 0.5)), (0, 255, 0), 3)

    # 
    img = getRegion(img, vertex)

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

    #cv2.imshow("Image", img)

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
        bottom_width = 180
    
        top_width = (top_width / 2) + 55
        bottom_width = (bottom_width / 2)

        max_width = img.shape[1]
        max_height = img.shape[0] - 60

        print(max_width)
        print(max_height)
        
        width_delta = int(max_width/10)

        bottom_left = (bottom_width, max_height)
        bottom_right = (max_width - bottom_width, max_height)
        
        top_right = (max_width / 2 + width_delta, max_height / 2 + top_width)
        top_left = (max_width / 2 - width_delta, max_height / 2 + top_width)

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

    rho = 2
    theta = np.pi / 180
    thres = 10
    min_line_length = 10
    max_line_gap = 80

    hough = cv2.HoughLinesP(img, rho, theta, thres, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

    if(hough is None or len(hough) > 500):
        return []
    else:
        return hough


def arcTan2(y, x):

    if y == 0.0:
        return 0.0
    elif x == 0.0:
        return 90.0

    a = min(math.fabs(x), math.fabs(y)) / min(math.fabs(x), math.fabs(y))
    s = (a * a)

    angle = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a

    #print(s)

    if math.fabs(y) > math.fabs(x):
        angle = (math.pi / 2)

    if x < 0:
        angle = math.pi - angle
        
    if y < 0:
        angle *= - 1.0

    angle *= math.pi / 180

    if angle < 0:
        angle += 180
        
    return angle

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

            #if y < 280:
            #   break

            #ngle = arcTan2((y2 - y1), (x2 - x1))
            angle = math.atan2((y2 - y1), (x2 - x1)) * 180 / math.pi

            if math.fabs(angle) <= 10:
                continue
            
            if x == 0:
                dx = 1

            if x2 == x1:
                continue

            #print("angle = " + str(angle))

            #print(angle)
 
            if angle < 0.0:
                angle += 180

            #print(angle)

            #current_slope = (y2 - y1) / (x2 - x1)
            #intercept = y1 - current_slope * x1

            #x_min = x1
            #y_min = y1
            #x_max = x2
            #y_max = y2

            lines_list.append((line, angle, center))
            
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

        evaluatedLine = evaluateLine(line, img, original)
        
        for eLItem in evaluatedLine:
            
            (x, y) = eLItem[2]
            angle = eLItem[1]

            if x < img.shape[1] * 0.5 and angle >= 25 and angle <= 75:
                leftLines.append(eLItem)
                #print("left: " + str(angle))
                
            if x > img.shape[1] * 0.5and angle >= 105 and angle <= 155:
                rightLines.append(eLItem)
                #print("right: " + str(angle))


    for ll in leftLines:
        line = ll[0][0]
        cv2.line(original, (line[0], line[1]), (line[2], line[3]), (0, 0, 0), 3)


    for rl in rightLines:
        line = rl[0][0]
        cv2.line(original, (line[0], line[1]), (line[2], line[3]), (0, 0, 0), 3)


   # processLine(rightLines, original, True)
    #processLine(leftLines, original, False)
    
      
    for ll in leftLines:
        for rl in rightLines:

            ll_x, ll_y = ll[2]            
            #cv2.circle(original, (math.ceil(ll_x), math.ceil(ll_y)), 4, (255,0,0), 4)

            rl_x, rl_y = rl[2]
            #cv2.circle(original, (math.ceil(rl_x), math.ceil(rl_y)), 4, (0,0,255), 2)


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
        img = cv2.imread("image/test.jpg", 1)
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
        

    
