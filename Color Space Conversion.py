import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取Lena图像
image = cv2.imread("1.bmp")

# (1) RGB到CMY的转换
# CMY = 1 - RGB
c = 1 - image[:,:,2] / 255.0
m = 1 - image[:,:,1] / 255.0
y = 1 - image[:,:,0] / 255.0

# 将CMY图像组合到一个多通道图像中
cmy_image = cv2.merge([c*255, m*255, y*255])
cmy_image = np.uint8(cmy_image)

# (1) RGB到HSI的转换

def rgb2hsi(rgb):
    r, g, b = rgb[:,:,0]/255., rgb[:,:,1]/255., rgb[:,:,2]/255.
    intensity = (r + g + b) / 3.
    min_val = 0.000001
    denominator = np.sqrt((r-g)**2 + (r-b)*(g-b))
    theta = np.arccos(0.5 * ((r-g)+(r-b)) / np.maximum(min_val, np.clip(denominator, -1, 1)))
    h = theta
    h[b>g] = 2*np.pi - h[b>g]
    h /= 2 * np.pi
    s = 1 - 3.0 * np.minimum(r, np.minimum(g, b)) / (r+g+b + min_val)
    return cv2.merge([np.clip(h, 0, 1)*255, np.clip(s, 0, 1)*255, intensity*255]).astype(np.uint8)

hsi_image = rgb2hsi(image)



# (1) RGB到LAB的转换
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# 使用plt同屏显示图像
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('RGB')
plt.axis('off')  # 关闭坐标轴

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(cmy_image, cv2.COLOR_BGR2RGB))
plt.title('CMY')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(hsi_image, cv2.COLOR_BGR2RGB))
plt.title('HSI')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(lab_image, cv2.COLOR_BGR2RGB))
plt.title('LAB')
plt.axis('off')

plt.tight_layout()
plt.show()

# (3) 保存各个图像到本地
cv2.imwrite('CMY_Image.tif', cmy_image)
cv2.imwrite('HSI_Image.tif', hsi_image)
cv2.imwrite('LAB_Image.tif', lab_image)
