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

    #cv2.imshow("Yellow", img) 

    # compute region of intrest
    vertex = computeROI(img)
    # 
    img = getRegion(img, vertex)

    #cv2.imshow("Regio", img)

    # convert bgr2gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # segment detection
    LineSegmentDetector = cv2.createLineSegmentDetector()
    segmentDetectRes = LineSegmentDetector.detect(img)
    #segmentDetectImg = LineSegmentDetector.drawSegments(original, segmentDetectRes[0])

    drawLines(img, original, segmentDetectRes[0], vertex)

    # assistence lines
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
    
    #cv2.imshow("cvtColor", img)

    # print original
    #cv2.imshow("Original", segmentDetectImg)

    cv2.imshow("Original", original)    

def evaluateLine(line, img, original):
    
    #lines = houghLines(img)
    #lines_original = np.zeros(img.shape, dtype=np.uint8)
    #line_width = 2

    width = img.shape[0]
    height = img.shape[1]
    
    lines_list = []

    #color = (0, 0, 255)

    try:
        for x1, y1, x2, y2 in np.array(line):
            
            y1 = -(y1 - height) 
            y2 = -(y2 - height) 

            x = (x1 + x2) / 2
            y = (y1 + y2) / 2

            current_slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - current_slope * x1
            
            center = (x, y)
            angle = math.atan2((y2 - y1), (x2 - x1)) * 180 / math.pi

            if math.fabs(angle) <= 10:
                continue
            
            if x == 0:
                dx = 1

            if x2 == x1:
                continue

            if angle < 0.0:
                angle += 180

            lines_list.append((line, angle, current_slope, intercept))
            
    except TypeError as e:
        print("Type Error => " + str(e) )

    return lines_list


def calc_avg_slope(lines, old_avg_slope, old_avg_bias, discount_rate = 0.5):

    def get_average(old, new, discount_rate):
        if old == 0:
            return new
        return (1 - discount_rate) * old + discount_rate * new

    if len(lines) > 0:
        lines, angle, slope, bias = np.mean(lines, 0)
        new_slope = get_average(old_avg, slope, discount_rate)
        new_bias = get_average(old_avg, bias, discount_rate)
        return new_slope, new_bias
    return 0,0

def drawLines(img, original, lines, vertex):

    lines_original = np.zeros(img.shape, dtype=np.uint8)
    line_width = 2

    color = (0, 0, 255)

    # sortlines

    leftLines = []
    rightLines = []

    print(vertex)

    for line in lines:
        for eLItem in evaluateLine(line, img, original):

            intercept = eLItem[3]
            slope = eLItem[2]
            angle = eLItem[1]
            line = np.array(eLItem[0])

            print(slope)

            x = line[0][0]
            y = line[0][1]

            # calculate roi x center
            roi_p1_x = vertex[0][0][0]
            roi_p2_x = vertex[0][1][0]
            midx = ((roi_p2_x-roi_p1_x) / 2) + roi_p1_x
            
            if x < midx and angle >= 25 and angle <= 55 and slope > 0:
                leftLines.append(eLItem)
                
                print("left = " + str(midx) + " < " + str(x) +  " == " +  str(slope))

            if x >= midx and angle >= 125 and angle <= 155 and slope < 0:
                rightLines.append(eLItem)

                print("right = " + str(midx) + " > " + str(x) + " == " + str(slope))


    #lines, angle, slope, intercept = calc_avg_slope(leftLines, )

    for ll in leftLines:

        line_coor = ll[0][0]

        p1_x = line_coor[0]
        p1_y = line_coor[1]
        p2_x = line_coor[2]
        p2_y = line_coor[3]

        cv2.line(original, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 3)

    #print(rightLines)

    for rl in rightLines:

        line_coor = rl[0][0]

        p1_x = line_coor[0]
        p1_y = line_coor[1]
        p2_x = line_coor[2]
        p2_y = line_coor[3]

        cv2.line(original, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 3)

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
        

    
