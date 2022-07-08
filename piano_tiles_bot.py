import cv2 as cv
import numpy as np
import pyautogui
from PIL import ImageGrab
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import pyautogui, sys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By


##BUGLAR
#BAZEN SATIR ÇOK KAYABİLİYOR
#VİDEOYU İZLE DİYİP TEKRAR BAŞLAYINCA X İ 1DEN FAZLA KEZ ARTTIRIYOR.

##GELİŞTİRMEK İSTERSEN:
#TAM EKRAN MODUNA GÖRE AYARLAYARAK DAHA KESİN SONUÇ ELDE EDEBİLİRSİN.
#ZAMANA GÖRE KAÇ PIXEL ARTACAĞI DAHA İYİ AYARLANMALI
#BELİRLİ DÜĞMELER ATAYARAK PROCESSINGE NE ZAMAN BAŞLAYI NE ZAMAN DURMASI GEREKTİĞİNİ KODA EKLEYEBİLİRSİN.
#2 FARKLI MASKING(BİR ÜST BİR ALT PİKSELLERDE) YAPARAK DAHA HIZLI SONUÇ ELDE EDEBİLİRSİN(BİRAZ BOŞ OLABİLİR,ÖNEC ÜSTTEKİNİ DENE)

x=0
y=450

s=Service(ChromeDriverManager().install())

black=np.uint8([[[0,2,3]]])
hsv_black=cv.cvtColor(black,cv.COLOR_BGR2HSV)
print(hsv_black)
path="C:\Program Files (x86)\chromedriver.exe"
#driver=webdriver.Chrome(path)
driver = webdriver.Chrome(service=s)

#driver.get("https://poki.com/en/g/piano-tiles-2")
driver.get("https://gaamess.com/piano-tiles-2/")
actionChain = webdriver.ActionChains(driver)
time.sleep(1)


while True:
    #screen=np.array(ImageGrab.grab(bbox=(625,590,950,600)))
    screen=np.array(ImageGrab.grab(bbox=(435,450,630,451)))
    hsv_screen=cv.cvtColor(screen,cv.COLOR_BGR2HSV)
    masked_b=cv.inRange(hsv_screen,np.array([20,255,3]),np.array([20,255,3]))
    masked=cv.inRange(hsv_screen,np.array([0,0,1]),np.array([0,0,1]))
    masked_combined=cv.bitwise_or(masked_b,masked)

    button1=np.zeros(screen.shape[:2],dtype="uint8")
    button1_rectangle=cv.rectangle(button1,(0,0),(47,330),255,-1)
    button1_last=cv.bitwise_and(masked_combined,masked_combined,mask=button1_rectangle)

    button2=np.zeros(screen.shape[:2],dtype="uint8")
    button2_rectangle=cv.rectangle(button2,(49,0),(97,330),255,-1)
    button2_last=cv.bitwise_and(masked_combined,masked_combined,mask=button2_rectangle)

    button3=np.zeros(screen.shape[:2],dtype="uint8")
    button3_rectangle=cv.rectangle(button3,(98,0),(146,330),255,-1)
    button3_last=cv.bitwise_and(masked_combined,masked_combined,mask=button3_rectangle)

    button4=np.zeros(screen.shape[:2],dtype="uint8")
    button4_rectangle=cv.rectangle(button4,(148,0),(195,330),255,-1)
    button4_last=cv.bitwise_and(masked_combined,masked_combined,mask=button4_rectangle)
    #cv.imshow("deneme",masked)
    button1_white=np.sum(button1_last==255)
    button2_white=np.sum(button2_last==255)
    button3_white=np.sum(button3_last==255)
    button4_white=np.sum(button4_last==255)

    
    if x>=200:
        if x%25==0:
            y+=2

    if button1_white>20:
        pyautogui.moveTo(445, y,duration=0)           
        pyautogui.click()
        print("d")
        print(x)
        x+=1
        
    if button2_white>20:
        pyautogui.moveTo(502, y) 
        pyautogui.click() 
        print("f")
        print(x)
        x+=1
        
    if button3_white>20:
        pyautogui.moveTo(570, y)
        pyautogui.click()  
        print("j")
        print(x)
        x+=1
        
    if button4_white>20:
        pyautogui.moveTo(625, y) 
        pyautogui.click() 
        print("k")
        print(x)
        x+=1
        
    

    


    cv.imshow("deneme2",button1_last)
    cv.imshow("deneme3",button2_last)
    cv.imshow("deneme4",button3_last)
    cv.imshow("deneme5",button4_last)
    cv.imshow("asdcsac",masked_combined)
    cv.imshow("asca",masked_b)
    cv.imshow("ekran",screen)
    cv.waitKey(1)
