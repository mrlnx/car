import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from time import sleep
import csv
import random

hlines = False
debugging = False

def main(img):
    
    original = img

    # convert yellow colors to white
    img = cvt_yellow(img)

    # compute region of intrest
    vertex = compute_roi(img)

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
            cv2.circle(original, (p2_x, p2_y), 4, (255,0,), 4)
    # 
    img = get_region(img, vertex)

    # convert bgr2gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur
    size = 7
    img = cv2.GaussianBlur(img, (size, size), 0)

    # canny edge detection
    low_thres = 100
    high_thres = 300

    img = cv2.Canny(img, low_thres, high_thres)

    #merge original with lines
    edges = np.dstack((img, img, img))
    test = cv2.addWeighted(edges, 2, original, 1, 0)

    draw_lines(img, original)

    #1280x720
    cv2.imshow("Original", original)

def cvt_yellow(img):
    
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

def compute_roi(img):

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

def get_region(img, vertex):

    mask = np.zeros_like(img)
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertex, ignore_mask_color)


    return cv2.bitwise_and(img, mask)


def hough_lines(img):

    rho = 0.8
    theta = np.pi / 180
    thres = 25
    min_line_length = 50
    max_line_gap = 200

    hough = cv2.HoughLinesP(img, rho, theta, thres, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

    if(hough is None or len(hough) > 500):
        return [0,0,0,0]
    else:
        return hough
    
def evaluate_line(line, img, original):
    
    lines = hough_lines(img)

    # black
    lines_original = np.zeros(img.shape, dtype=np.uint8)

    width = img.shape[0]
    height = img.shape[1]
    
    lines_list = []
    
    try:
        
        for x1, y1, x2, y2 in line:

            old_x = (x1 + x2) / 2
            old_y = (y1 + y2) / 2

            y1 = -(y1 - height) 
            y2 = -(y2 - height) 
            
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2

            center = (int(old_x), int(old_y))
            angle = math.atan2((y2 - y1), (x2 - x1)) * 180 / math.pi

            if math.fabs(angle) <= 10:
                continue
            
            if x == 0:
                x = 1

            if x2 == x1:
                continue

            if angle < 0.0:
                angle += 180

            current_slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - current_slope * x1
    
            lines_list.append([line, angle, center])
            
    except TypeError as e:
        print_value("Type Error = ", e)

    return lines_list

def get_slope(x1, y1, x2, y2):

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - current_slope * x1

    return slope, intercept

def print_value(key, value):
    if debugging:
        print(key, str(value))

def draw_lines(img, original):

    lines = hough_lines(img)
    lines_original = np.zeros((*img.shape, 3), dtype=np.uint8)
    line_width = 2

    ymin = img.shape[0]
    ymax = img.shape[0]

    data = dict()

    # sortlines
    left_lines = []
    right_lines = []

    all_left_grad = []
    all_left_y = []
    all_left_x = []

    all_right_grad = []
    all_right_y = []
    all_right_x = []

    for line in lines:
        for eval_item in evaluate_line(line, img, original):
            
            (x, y) = eval_item[2]
            angle = eval_item[1]
            coor = eval_item[0][0]

            x1 = coor[0]
            y1 = coor[1]
            x2 = coor[2]
            y2 = coor[3]

            ymin = min(min(y1, y2), ymin)
            gradient, intercept = np.polyfit((x1, x2), (y1, y2), 1)

            # eliminate lines
            if x < img.shape[1] * 0.5 and angle >= 25 and angle <= 55 and gradient < 0:

                left_lines.append(eval_item)

                all_left_grad += [gradient]
                all_left_y += [y1, y2]
                all_left_x += [x1, x2]
            
            if x > img.shape[1] * 0.5 and angle >= 125 and angle <= 155 and gradient > 0:

                right_lines.append(eval_item)

                all_right_grad += [gradient]
                all_right_y += [y1, y2]
                all_right_x += [x1, x2]

    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_mean_grad * left_x_mean)

    print_value("all line grad = ", gradient)
    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)

    if len(all_left_grad) > 0 and len(all_right_grad) > 0:

        upper_left_x = int((ymin - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax - left_intercept) / left_mean_grad)

        upper_right_x = int((ymin - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax - right_intercept) / right_mean_grad)

        p1_center_x = int((upper_left_x + upper_right_x) / 2)
        p2_center_x = int((lower_left_x + lower_right_x) / 2)

        angle = math.atan2((p1_center_x - ymin), (p2_center_x - ymax)) * 180 / math.pi

        print("Deviation: " + str(-float(angle - 90)))

        tl = (upper_right_x, ymin)
        tr = (upper_left_x, ymin)
        br = (lower_left_x, ymax)
        bl = (lower_right_x, ymax)

        cv2.fillPoly(original, np.array([[tl, tr, br, bl]], np.int32), (0,255,0))
        cv2.line(original, (upper_left_x, ymin), (lower_left_x, ymax), (0, 0, 255), 4)
        cv2.line(original, (upper_right_x, ymin), (lower_right_x, ymax), (0, 0, 255), 4)
        cv2.line(original, (p1_center_x, ymin), (p2_center_x, ymax), (0, 0, 255), 2)

        (transform, w, h) = transform_lane(tl, tr, br, bl)

        warped = cv2.warpPerspective(original, transform, (w,h))

        cv2.imshow("Warped", warped)
        

def order_pts(pts):

    rect = np.zeros((4,2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmin(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(s)]
    rect[3] = pts[np.argmin(s)]

    return rect

def transform_lane(tl, tr, br, bl):

    rect = np.array([tl, tr, br, bl], dtype= "float32")

    (tl, tr, br, bl) = rect

    w1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    w2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(w1), int(w2))
    
    h1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    h2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(h1), int(h2))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    print(dst)
    
    transform = cv2.getPerspectiveTransform(rect, dst)

    return transform, maxWidth, maxHeight
    

if __name__ == "__main__":

    content_type = "image"

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

            main(img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        capture.release()
        cv2.destroyAllWindows()
