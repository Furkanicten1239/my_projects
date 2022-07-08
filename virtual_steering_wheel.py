import cv2 as cv
import numpy as np
import time
from pynput.keyboard import Key, Controller

keyboard = Controller()
webcam=cv.VideoCapture(0)

yellow1=np.uint8([[[212,145,71]]])
yellow2=np.uint8([[[0,255,0]]])
hsv_yellow1=cv.cvtColor(yellow1,cv.COLOR_BGR2HSV)
hsv_yellow2=cv.cvtColor(yellow2,cv.COLOR_BGR2HSV)
print(hsv_yellow1)
#print(hsv_yellow2)

while True:
    _,aaa=webcam.read()
    frame=cv.flip(aaa,1)
    hsv_frame=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    #masked=cv.inRange(hsv_frame,np.array([19,100,100]),np.array([39,255,255]))
    masked=cv.inRange(hsv_frame,np.array([94,100,100]),np.array([114,255,255]))
    masked_cleaned = np.zeros(masked.shape,np.uint8)
    contours, hier=cv.findContours(masked,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 200<cv.contourArea(cnt)<50000:
            cv.drawContours(frame,[cnt],0,(0,0,0))
            cv.drawContours((masked_cleaned),[cnt],0,255,-1)

    contours2, hier2=cv.findContours(masked_cleaned,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

    coordinates=cv.HoughLinesP(masked_cleaned,1,np.pi/180,50,maxLineGap=0,minLineLength=100)
    
    try:
        for lines in coordinates:
            x1,y1,x2,y2=lines[0]
            cv.line(frame,(x1,y1),(x2,y2),(0,0,255),3)
            uzunluk=((y2-y1)**2+(x2-x1)**2)**(1/2)
            #print("uzunluk= ",uzunluk)
            tangent=((y2-y1)/(x2-x1))
            print(tangent)
            

        if tangent<-0.07:
            keyboard.press("w")
            keyboard.press("a")
            keyboard.release("d")
        elif tangent>0.07:
            keyboard.press("w")
            keyboard.press("d")
            keyboard.release("a")
        elif 0<tangent<0.07 or -0.07<tangent<0:
            keyboard.press("w")
            keyboard.release("a")
            keyboard.release("d")
        
    except:
        keyboard.release("a")
        keyboard.release("d")
        keyboard.release("w")

    #cv.imshow("as",masked)
    cv.imshow("asdc",frame)
    #cv.imshow("bitwise not",masked_cleaned)
    cv.waitKey(1)
