import numpy as np
import cv2
from PIL import Image,ImageEnhance 
import matplotlib.pyplot as plt
"""
hough变换是一种常用的计算机视觉图形检验方法，霍夫变换一般用于检验直线或者圆。

"""
img = Image.open(r"digital signal processing/直线变换.png")
#增强图像效果
img = ImageEnhance.Contrast(img).enhance(3)
img.show()
#处理成矩阵，便于后续处理
img = np.array(img)
#灰度处理
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.THRESH_OTSU具有双峰值，显示效果更好.
"""
cv2.THRESH_OTSU使用最小二乘法处理像素点。一般情况下,cv2.THRESH_OTSU适合双峰图。
cv2.THRESH_TRIANGLE使用三角算法处理像素点。一般情况下,cv2.THRESH_TRIANGLE适合单峰图。
"""
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
#canny边缘检验算法处理
result = cv2.Canny(thresh, ret-30, ret+30, apertureSize=3)

#霍夫变换检测直线
lines = cv2.HoughLinesP(result, 1, 1 * np.pi / 180, 10, minLineLength=10, maxLineGap=5)
# 画出检测的线段
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0),2)
img = Image.fromarray(img, 'RGB')
img.show()
