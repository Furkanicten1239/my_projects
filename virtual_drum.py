import cv2 as cv
import numpy as np
import pygame as py
import time

py.mixer.init()


webcam=cv.VideoCapture(0)

#60,68,2
green=np.uint8([[[60,68,2]]])
hsv_green=cv.cvtColor(green,cv.COLOR_BGR2HSV)
#print(hsv_green)

red=np.uint8([[[27,46,171]]])
hsv_red=cv.cvtColor(red,cv.COLOR_BGR2HSV)
#print(hsv_red)

while True:
    _,abc=webcam.read()
    frame=cv.flip(abc,1)#flipping webcam to move more easily(at my side while using pencils)

    #CERTAIN FRAMES

    drum_green=np.zeros(frame.shape[:2],dtype="uint8")
    drum_red=np.zeros(frame.shape[:2],dtype="uint8")
    green_frame=cv.rectangle(drum_green,(0,0),(100,100),255,thickness=-1)
    red_frame=cv.rectangle(drum_red,(640,0),(540,100),255,thickness=-1)
    drum1_green=cv.bitwise_and(frame,frame,mask=green_frame)
    drum1_red=cv.bitwise_and(frame,frame,mask=red_frame)

    #CONVERTING

    drum1_hsv=cv.cvtColor(drum1_green,cv.COLOR_BGR2HSV)
    drum2_hsv=cv.cvtColor(drum1_red,cv.COLOR_BGR2HSV)
    hsv_frame=cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    #MASKING

    masked_green=cv.inRange(drum1_hsv,np.array([76,100,50]),np.array([96,255,255]))
    masked_red=cv.inRange(drum2_hsv,np.array([3,150,150]),np.array([14,255,255]))

    contours_green, hier_green = cv.findContours(masked_green,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    contours_red, hier_red = cv.findContours(masked_green,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

    #ALGORITHM PART

    amount_of_green=np.sum(masked_green==255)
    amount_of_red=np.sum(masked_red==255)

    if amount_of_green>0:
        print("drum1111111111111")
        py.mixer.music.load("C:/Users/gazi/Desktop/kodlama/opencv_projects/music/drum1.mp3")
        py.mixer.music.play()
        
    
    if amount_of_red>0:
        print("drum2222222222")
        py.mixer.music.load("C:/Users/gazi/Desktop/kodlama/opencv_projects/music/drum2.mp3")
        py.mixer.music.play()
        

    #DRAWING RECTANGLES
    cv.rectangle(frame,(0,0),(100,100),(255,0,0),2)
    cv.rectangle(frame,(640,0),(540,100),(0,255,0),2)
    ####
    cv.imshow("webcam",frame)
    cv.imshow("green",masked_green)
    cv.imshow("red",masked_red)
    


    cv.waitKey(1)