import cv2
import numpy as np
from matplotlib import pyplot as plt

# 加载图像
file_path = 'flower.png'
image = cv2.imread(file_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 转换到HSV色彩空间
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 设置红色的阈值范围
# 注意：红色在HSV空间中有两个范围，因为它跨越了0度
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# 根据阈值创建掩码
mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
mask = mask1 + mask2

# 将掩码与原图像结合，只保留红色区域
red_only = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# 弱化背景区域，通过减少掩码区域外的饱和度和明度
background = cv2.bitwise_and(image_rgb, image_rgb, mask=cv2.bitwise_not(mask))
background = cv2.cvtColor(background, cv2.COLOR_RGB2HSV)
background[..., 1] = background[..., 1] // 3
background[..., 2] = background[..., 2] // 3
background = cv2.cvtColor(background, cv2.COLOR_HSV2RGB)
red_flower_black_bg = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)


# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(red_flower_black_bg)
plt.title('Red Color Extracted')
plt.axis('off')

plt.show()
