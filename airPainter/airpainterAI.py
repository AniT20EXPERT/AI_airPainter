import os
import cv2
import time
import mediapipe as mp
import numpy as np
import geminiapi
import threading
import handtrackingmodule as htm

detector = htm.handDetector(detectionCon=0.85)
pTime = 0
myList = os.listdir("toolsbanners")
overlaylist = []

#####################
brushThickness = 18
eraserThickness = 100
#####################



for imgpath in myList:
    image = cv2.imread(f"toolsbanners/{imgpath}")
    image = cv2.resize(image, (720, 133))
    overlaylist.append(image)

#print(len(overlaylist))

header = overlaylist[0]

cap = cv2.VideoCapture(0)
#setting capture device resolution:-
cap.set(3, 1280)
cap.set(4, 720)

img_canvas = np.zeros((720, 1280, 3), np.uint8)



tiplist = [8, 12, 16, 20]

circle_color = (0,0,255)

xp, yp = 0, 0

#initializing timer
last_save_time = time.time()

while True:
    #importing image from cam
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    # Check 8 seconds have passed since the last frame was saved
    current_time = time.time()
    if current_time - last_save_time >= 8:
        # Save the frame
        cv2.imwrite("img_to_process.jpg", img_canvas)
        # Update the last save time
        last_save_time = current_time
        # Create a new thread to handle the Gemini API call
        threading.Thread(target=geminiapi.printResponse, args=("img_to_process.jpg",)).start()

    #finding the landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        #tip of index finger
        x1,y1 = lmList[8][1:]
        #tip of middle finger
        x2,y2 = lmList[12][1:]

        # check which finger is up

        fingersclosed = []
        # lmList[landmark id][id,x,y]
        # check other fingers
        for i in tiplist:
            if i == 4:
                continue
            if lmList[i][2] > lmList[i - 2][2]:
                fingersclosed.append(0)
            else:
                fingersclosed.append(1)

        totalfingers = fingersclosed.count(1)

        #print(totalfingers)

    #for two finger up we go to selection mode
        if totalfingers==2:
            cv2.rectangle(img, (x1,y1-15), (x2,y2), (0,0,0), cv2.FILLED)
            xp,yp = 0, 0
            # print("selection mode")
            #checking selection
            if y1 <133:
                if 460 > x1 > 280:
                    header = overlaylist[0]
                    circle_color = (0,0,255)
                elif 640 > x1 > 460:
                    header = overlaylist[1]
                    circle_color = (255, 0, 0)
                elif 820 > x1 > 640:
                    header = overlaylist[2]
                    circle_color = (0, 255, 0)
                elif 1000 > x1 > 820:
                    header = overlaylist[3]
                    circle_color = (255, 255, 255)

        #print(circle_color)
    #check if index is up so we go to draw mode
        if totalfingers == 1:
            if circle_color == (255, 255, 255):
                cv2.circle(img, (x1,y1), eraserThickness//2, circle_color, cv2.FILLED)
            else:
                cv2.circle(img, (x1,y1), 15, circle_color, cv2.FILLED)

            # print("drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if circle_color == (255, 255, 255):
                cv2.line(img_canvas, (xp, yp), (x1, y1), (0, 0, 0), eraserThickness)
            else:
                cv2.line(img_canvas, (xp, yp), (x1,y1), circle_color, brushThickness)
            xp, yp = x1, y1
        if totalfingers == 4:
            img_canvas[:] = (0, 0, 0)

    imgGray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, img_canvas)


    #setting up the header
    img[0:133, 280:1000] = header
    cv2.putText(img, "Hold 4 fingers to reset drawing", (370, 160), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 0, 0), 2)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("image", img)
    #cv2.imshow("canvas", img_canvas)
    cv2.waitKey(1)
