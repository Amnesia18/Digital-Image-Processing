import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# 读取图像
image = cv2.imread('Fig0304(a)(breast_digital_Xray).tif', cv2.IMREAD_GRAYSCALE)

# 计算直方图
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# 将直方图归一化到范围 [0, 1]
hist /= hist.sum()

# 创建直方图图表
plt.figure(figsize=(8, 6))
plt.plot(hist, color='blue', label='Histogram')

# 计算核密度估计 (KDE)
pixel_values = np.arange(256)
kde = gaussian_kde(image.ravel())
kde_values = kde(pixel_values) / kde(pixel_values).sum()

# 添加核密度估计曲线
plt.plot(kde_values, color='red', label='KDE')

# 添加标签和标题
plt.xlabel('像素值')
plt.ylabel('归一化频率')
plt.title('Histogram with KDE for Breast Image')

# 显示图例
plt.legend()

# 显示直方图和KDE图
plt.show()
