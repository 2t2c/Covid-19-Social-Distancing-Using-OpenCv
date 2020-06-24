import cv2
import imutils
import numpy as np
from imutils import contours, perspective
from scipy.spatial import distance as dist



cap = cv2.VideoCapture("vtest.avi")
fgbg = cv2.createBackgroundSubtractorMOG2()

ret, frame = cap.read()

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Gaussian Blur
        gblur = cv2.GaussianBlur(img_gray, (5,5), 0)
        fg_mask = fgbg.apply(gblur)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closing = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(opening, kernel, iterations=3)
        # removes shadows
        retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(bins, None, iterations = 3)

        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(frame1, contours, -1, (100,46,200), 2)
        minarea = 1500
        maxarea = 50000
        cx = np.zeros(len(contours))
        cy = np.zeros(len(contours))
        for i in range(len(contours)):
            # IMPORTANT
            # using hierarchy to only count parent contours (contours not within others)
            # if hierarchy[0, i, 3] == -1:
            area = cv2.contourArea(contours[i])
            if minarea < area < maxarea:

                cnt = contours[i]
                m = cv2.moments(cnt)

                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])

                x, y, w, h = cv2.boundingRect(contours[i])

                if area > 4500:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (100,46,200), 2)
                    cv2.putText(frame, "WARNING", (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(100, 46, 200),2)
                else:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (10,210,10), 2)
                    cv2.putText(frame, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, .3,
                            (100,46,200), 1)
                cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, markerSize=4, thickness=3,
                               line_type=cv2.LINE_8)

        cv2.imshow("Social Distancing", frame)

        if cv2.waitKey(60) == 27:
            break

    else: break

# (100,46,200)

cv2.destroyAllWindows()
cap.release()