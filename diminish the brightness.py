import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取Lena图像
image = cv2.imread("digital signal processing/lena.tif")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB

# (1) 将该RGB图像的亮度降低70％
brightness_decreased_image = np.clip(image_rgb * 0.3, 0, 255).astype(np.uint8)

# (2) 将该RGB图像中的红色分量的亮度降低50％
red_decreased_image = image_rgb.copy()
red_decreased_image[:,:,0] = np.clip(red_decreased_image[:,:,0] * 0.5, 0, 255)

# (3) 将该RGB图像中的绿色分量的亮度降低50％
green_decreased_image = image_rgb.copy()
green_decreased_image[:,:,1] = np.clip(green_decreased_image[:,:,1] * 0.5, 0, 255)

# 使用matplotlib同屏显示图像
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(brightness_decreased_image)
plt.title('Brightness Decreased by 70%')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(red_decreased_image)
plt.title('Red Channel Decreased by 50%')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(green_decreased_image)
plt.title('Green Channel Decreased by 50%')
plt.axis('off')

plt.tight_layout()
plt.show()
