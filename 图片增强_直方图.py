import cv2
import matplotlib.pyplot as plt

# 读取图像
pollen_image = cv2.imread('Fig0304(a)(breast_digital_Xray).tif', cv2.IMREAD_GRAYSCALE)
lena_image = cv2.imread('lena.tif', cv2.IMREAD_GRAYSCALE)

# 进行直方图均衡
pollen_equalized = cv2.equalizeHist(pollen_image)
lena_equalized = cv2.equalizeHist(lena_image)

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 显示原始图像
axes[0, 0].imshow(pollen_image, cmap='gray')
axes[0, 0].set_title('Pollen Image (Original)')
axes[0, 0].axis('off')

axes[0, 1].imshow(lena_image, cmap='gray')
axes[0, 1].set_title('Lena Image (Original)')
axes[0, 1].axis('off')

# 显示增强后的图像
axes[1, 0].imshow(pollen_equalized, cmap='gray')
axes[1, 0].set_title('Pollen Image (Equalized)')
axes[1, 0].axis('off')

axes[1, 1].imshow(lena_equalized, cmap='gray')
axes[1, 1].set_title('Lena Image (Equalized)')
axes[1, 1].axis('off')

# 绘制增强前后的直方图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(pollen_image.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.6)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of Pollen Image (Original)')

plt.subplot(1, 2, 2)
plt.hist(lena_image.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.6)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of Lena Image (Original)')

plt.tight_layout()
plt.show()
