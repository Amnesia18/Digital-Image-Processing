import cv2
import numpy as np
from matplotlib import pyplot as plt

# 加载带有模糊和噪声的图像
noisy_blurred_image = cv2.imread("digital signal processing/checkerboard.jpg", cv2.IMREAD_GRAYSCALE)

# 执行直接逆滤波的函数
def direct_inverse_deblur(image, kernel):
    # 执行逆滤波
    deblurred_image = cv2.filter2D(image, -1, np.linalg.pinv(kernel))
    return deblurred_image

# 定义模糊核（您可能需要根据实际情况进行调整）
kernel_size = 7
kernel = np.zeros((kernel_size, kernel_size))
kernel[kernel_size // 2, :] = 1.0 / kernel_size

# 执行直接逆滤波
direct_inverse_deblurred_image = direct_inverse_deblur(noisy_blurred_image, kernel)

# 执行Wiener滤波的函数
def wiener_deblur(image, kernel, noise_variance):
    # 计算核的功率谱
    kernel_fft = np.fft.fft2(kernel, image.shape)
    kernel_power_spectrum = np.abs(kernel_fft) ** 2

    # 计算图像的功率谱
    image_fft = np.fft.fft2(image)
    image_power_spectrum = np.abs(image_fft) ** 2

    # 执行Wiener去模糊
    deblurred_image_fft = np.conj(kernel_fft) / (kernel_power_spectrum + noise_variance)
    deblurred_image_fft *= image_fft
    deblurred_image = np.fft.ifft2(deblurred_image_fft).real
    return deblurred_image

# 定义噪声方差（您可能需要根据实际情况进行调整）
noise_variance = 25.0

# 执行Wiener去模糊
wiener_deblurred_image = wiener_deblur(noisy_blurred_image, kernel, noise_variance)

# 使用matplotlib显示结果
plt.figure(figsize=(12, 5))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为中文宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题

plt.subplot(131), plt.imshow(noisy_blurred_image, cmap='gray')
plt.title('带噪声和模糊的图像'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(direct_inverse_deblurred_image, cmap='gray')
plt.title('直接逆滤波'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(wiener_deblurred_image, cmap='gray')
plt.title('Wiener滤波'), plt.xticks([]), plt.yticks([])

plt.show()
