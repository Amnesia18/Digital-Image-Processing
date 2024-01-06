import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# 创建图像f1
image_size = 64
f1 = np.zeros((image_size, image_size))
bright_strip_width = 16
bright_strip_height = 40
f1[(image_size - bright_strip_height) // 2:(image_size + bright_strip_height) // 2, (image_size - bright_strip_width) // 2:(image_size + bright_strip_width) // 2] = 255

# 计算f1的傅里叶变换
F1 = np.fft.fft2(f1)
F1_magnitude = np.abs(np.fft.fftshift(F1))

# 创建图片f2
f2 = np.zeros((image_size, image_size))
for x in range(image_size):
    for y in range(image_size):
        f2[x, y] = (-1) ** (x + y) * f1[x, y]

# 计算f2的傅里叶变换
F2 = np.fft.fft2(f2)
F2_magnitude = np.abs(F2)


# 顺时针旋转f2得到f3
f3 = rotate(f2, 90, reshape=False)

# 计算f3的傅里叶变换
F3 = np.fft.fft2(f3)
F3_magnitude = np.abs(F3)

# 创建子图
fig, axes = plt.subplots(3, 2, figsize=(8, 12))
fig.suptitle('Image and FFT Comparison')

# 显示f1和FFT(f1)
axes[0, 0].imshow(f1, cmap='gray')
axes[0, 0].set_title('Image f1')
axes[0, 1].imshow(np.log(F1_magnitude + 1), cmap='gray')
axes[0, 1].set_title('FFT(f1) Magnitude Spectrum')

# 显示f2和FFT(f2)
axes[1, 0].imshow(f2, cmap='gray')
axes[1, 0].set_title('Image f2')
axes[1, 1].imshow(np.log(F2_magnitude + 1), cmap='gray')
axes[1, 1].set_title('FFT(f2) Magnitude Spectrum')

# 显示f3和FFT(f3)
axes[2, 0].imshow(f3, cmap='gray')
axes[2, 0].set_title('Image f3 (Rotated)')
axes[2, 1].imshow(np.log(F3_magnitude + 1), cmap='gray')
axes[2, 1].set_title('FFT(f3) Magnitude Spectrum')

# 隐藏坐标轴
for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
