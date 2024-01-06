import cv2
import numpy as np
 
img = cv2.imread("1.png")
imgGrey = cv2.imread("1.png",0)
 
sp1 = img.shape
sp2 = imgGrey.shape
 
mean_img_value = np.mean(img)
 
print(sp1)
print(sp2)
print(mean_img_value)


imgSize = img.size#像素总数目
print(imgSize)
 
ty = img.dtype#图像数据类型
print(ty)