import numpy as np
import cv2

# Input image
camera = cv2.VideoCapture(1)

while True:
    ret, image = camera.read()

    # Converts to grey for better reulsts
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Converts to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV values
    lower_skin = np.array([5,36,53])
    upper_skin = np.array([19,120,125])

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    cv2.imshow("Mask", mask)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Finds contours
    im2, cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draws contours
    for c in im2:
        if cv2.contourArea(c) < 3000:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 2)

        ## BEGIN - draw rotated rectangle
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,(0,191,255),2)
        ## END - draw rotated rectangle

    cv2.imshow("Feed", image)
    cv2.waitKey(1)
