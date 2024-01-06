from curses import COLS
import numpy as np
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
img=cv2.imread('1.png',cv2.IMREAD_UNCHANGED)#读取
# img=cv2.imread("1.png",0)#灰度


rows,cols = img.shape[:2]
# M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)# 原图像中心旋转90°
# 旋转变换
dst = cv.warpAffine(img,M,(COLS,rows)) #(cols,rows)画布大小 

cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
cv2.imshow('img',img)#显示
cv2.waitKey(0)