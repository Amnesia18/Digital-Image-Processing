    
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 在HSV空间对绿屏色彩区域进行阈值处理，生成遮罩进行抠图
img = cv.imread("flower.png", flags=1)  # 读取彩色图像
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 将图片转换到 HSV 色彩空间

# 使用 cv.inrange 函数在 HSV 空间检查设定的颜色区域范围，转换为二值图像，生成遮罩
lowerColor = np.array([0, 0, 0])  # (下限: 绿色33/43/46)
upperColor = np.array([180, 43, 220])  # (上限: 绿色77/255/255)
binary = cv.inRange(hsv, lowerColor, upperColor)  # 生成二值遮罩，指定背景颜色区域白色
binaryInv = cv.bitwise_not(binary)  # 生成逆遮罩，前景区域白色开窗，背景区域黑色
matting = cv.bitwise_and(img, img, mask=binaryInv)  # 生成抠图图像 (前景保留，背景黑色)

# 将背景颜色更换为红色: 修改逆遮罩 (抠图以外区域黑色)
imgReplace = img.copy()
imgReplace[binaryInv==0] = [0,0,255]  # 黑色背景区域(0/0/0) 修改为红色 (BGR:0/0/255)

plt.figure(figsize=(9, 6))
plt.subplot(221),plt.title("origin"), plt.axis('off')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.subplot(222), plt.title("binary mask"), plt.axis('off')
plt.imshow(binary, cmap='gray')
plt.subplot(223), plt.title("invert mask"), plt.axis('off')
plt.imshow(binaryInv, cmap='gray')
plt.subplot(224), plt.title("matting"), plt.axis('off')
plt.imshow(cv.cvtColor(matting, cv.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()
