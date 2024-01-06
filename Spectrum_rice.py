import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读入图像
image = cv2.imread("digital signal processing/rice.tif", cv2.IMREAD_GRAYSCALE)

# 计算图像的傅里叶变换
f_transform = np.fft.fft2(image)

# 将频谱原点移到中心
f_transform_centered = np.fft.fftshift(f_transform)

# 计算频谱的幅度
magnitude_spectrum = np.abs(f_transform_centered)

# 显示频谱
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(np.log(magnitude_spectrum + 1), cmap='gray')
plt.title('Centered FFT Magnitude Spectrum')
plt.colorbar()

plt.tight_layout()
plt.show()
