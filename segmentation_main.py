import cv2
import numpy as np
import keyboard

# Color Detection
def empty(a):
    pass

def stackImages(scale,imgArray):
    """
    Stack  tous les types d'image ensemble. Permet aussi de les scale
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
 
def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Trouve extreme outter* contour
    if not contours:
        return
    area = []
    for cnts in contours:
        area.append(cv2.contourArea(cnts))
    
    max_area = np.max(area)
    max_arg = np.argmax(area)
    cnt = contours[max_arg]

    #for cnt in contours:
    #    area = cv2.contourArea(cnt)
    #    if area > 200:
    cv2.drawContours(imgContour, cnt, -1, (255,0,0), 3) #-1: draw all contours
    peri = cv2.arcLength(cnt, True)
    print(peri)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True) # Approxime nombre de coins
    print(len(approx))
    objCorner = len(approx)
    x, y, w, h = cv2.boundingRect(approx)
    cv2.rectangle(imgContour, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(imgContour,"Area: {}".format(max_area),(x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0),2)
    return x,y,w,h

# Méthode qui fontionne le mieux!

def segmentation_contour(img):
    # Ajout d'un temps de repos
    imgContour = img.copy()
    imgFinal = img.copy()
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = 0
    h_max = 19
    s_min = 21
    s_max = 255
    v_min = 29
    v_max = 255
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgBlur = cv2.blur(mask,(5,5))
    imgBlur = cv2.morphologyEx(imgBlur, cv2.MORPH_CLOSE, (7,7))
    x, y, w, h = 0,0,0,0
    x, y, w, h = getContours(imgBlur, imgContour)
    cv2.imshow("Image finale", imgFinal[y:y+h, x:x+w])
    imgStack = stackImages(0.7, ([imgBlur], [imgContour]))
    cv2.imshow("Images stack", imgStack)

# Use Webcam
webcam = cv2.VideoCapture(0) # Seule caméra est celle de l'ordi
webcam.set(3,640) # id pour le nombre de pixel I guess 
webcam.set(4,480) # id pour le nombre de pixel I guess
webcam.set(10,75) # id pour le brightness
while True:
    sucess, img = webcam.read()
    segmentation_contour(img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break




##############################################################################################################################
##############################################################################################################################
#                                                     TESTS                                                                  #
##############################################################################################################################
##############################################################################################################################

## HSV values detection
#cv2.namedWindow("TrackBars")
#cv2.resizeWindow("TrackBars", (640,240))
#cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
#cv2.createTrackbar("Hue Max", "TrackBars", 33, 179, empty)
#cv2.createTrackbar("Saturation Min", "TrackBars", 133, 255, empty)
#cv2.createTrackbar("Saturation Max", "TrackBars", 217, 255, empty)
#cv2.createTrackbar("Value Min", "TrackBars", 173, 255, empty)
#cv2.createTrackbar("Value Max", "TrackBars", 255, 255, empty)

def segmentation_HSV():
    while True:
        sucess, img  = webcam.read()
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        #h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        #s_min = cv2.getTrackbarPos("Saturation Min", "TrackBars")
        #s_max = cv2.getTrackbarPos("Saturation Max", "TrackBars")
        #v_min = cv2.getTrackbarPos("Value Min", "TrackBars")
        #v_max = cv2.getTrackbarPos("Value Max", "TrackBars")
        h_min = 0
        h_max = 19
        s_min = 21
        s_max = 255
        v_min = 29
        v_max = 255
        #print(h_min, h_max, s_min, s_max, v_min, v_max)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask) # add 2 images ensemble et crée une seule
        imgResult = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)

        imgStack = stackImages(0.7, ([img, imgHSV],[mask, imgResult]))
        cv2.imshow("Images Stack", imgStack)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    return imgResult

#image_result = segmentation_HSV()
#cv2.imshow("image", image_result)
#cv2.waitKey(0)
####################################################################################################
bg = None
def run_avg(image, Weight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, Weight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    return thresholded

def segmentation_substract_background():
    count = True
    Weight = 0.5
    numFrames = 0
    global bg
    clone = webcam.read()
    while True:
        sucess, img  = webcam.read()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.blur(imgGray, (7,7),0)
        top, right, bottom, left = 10, 350, 225, 590
        if numFrames < 30:
            run_avg(imgBlur, Weight)
        else:
            # segment the hand region
            threshold = segment(imgBlur)
            threshold_petit = threshold[0:350,0:275]
            cv2.imshow("Thesholded", threshold)
            cv2.imshow("Thesholded_petit", threshold_petit)
        numFrames += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    return threshold_petit

#image_result = segmentation_substract_background()
#cv2.imshow("image", image_result)
#cv2.waitKey(0)

#########################################################################################################

def segmentation_contour_2():
    count = True
    Weight = 0.5
    numFrames = 0
    global bg
    clone = webcam.read()
    while True:
        sucess, img  = webcam.read()
        imgContour = img.copy()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.blur(imgGray, (3,3),0)
        top, right, bottom, left = 10, 350, 225, 590
        if numFrames < 30:
            run_avg(imgBlur, Weight)
        else:
            # segment the hand region
            threshold = segment(imgBlur)
            threshold_petit = threshold[0:350,0:275]
            imgCanny = cv2.Canny(threshold, 50,40)
            getContours(imgCanny, imgContour)
            imgStack = stackImages(0.7, ([imgCanny], [imgContour]))
            cv2.imshow("Images Stack", imgStack)
        numFrames += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def segmentation_contour_3():
    compte = 0
    while True:
        sucess, img = webcam.read()
        imgContour = img.copy()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.blur(imgGray, (3,3))
        imgBlur = cv2.medianBlur(imgBlur,3)
        ret, imgThreshold = cv2.threshold(imgBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imshow("threh", imgThreshold)
        imgCanny = cv2.Canny(imgThreshold, threshold1 = 300, threshold2 = 300)
        x,y,w,h = getContours(imgCanny, imgContour)
        imgStack = stackImages(0.7, ([imgThreshold], [imgContour]))
        cv2.imshow("Images Stack", imgStack)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

#segmentation_contour_2()
#segmentation_contour_3()
#Une ligne de trop pour modifier
#cv2.imshow("image", image_result)
#cv2.waitKey(0)