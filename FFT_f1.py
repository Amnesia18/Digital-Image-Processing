import numpy as np
import matplotlib.pyplot as plt

# # 创建图像f1
image_size = 64
f1 = np.zeros((image_size, image_size))
bright_strip_width = 16
bright_strip_height = 40
f1[(image_size - bright_strip_height) // 2:(image_size + bright_strip_height) // 2, (image_size - bright_strip_width) // 2:(image_size + bright_strip_width) // 2] = 255

# # 计算f1的傅里叶变换
F1 = np.fft.fft2(f1)
F1_magnitude = np.abs(np.fft.fftshift(F1))

# 显示原图f1
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.imshow(f1, cmap='gray')
plt.title('Original Image (f1)')

# 显示FFT(f1)的幅度谱图
plt.subplot(122)
plt.imshow(np.log(F1_magnitude+1 ), cmap='gray')
plt.title('FFT(f1) Magnitude Spectrum')
plt.colorbar()

plt.tight_layout()
plt.show()
