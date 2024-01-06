import cv2
import numpy as np
from matplotlib import pyplot as plt

# 加载原始图像
original_image = cv2.imread("digital signal processing/1.png", cv2.IMREAD_GRAYSCALE)

# 创建模糊核（根据需要调整）
kernel_size = 7
kernel = np.zeros((kernel_size, kernel_size))
kernel[kernel_size // 2, :] = 1.0 / kernel_size

# 使用模糊核进行模糊
blurred_image = cv2.filter2D(original_image, -1, kernel)

# 添加高斯噪声
noise_variance = 25.0  # 噪声方差（根据需要调整）
noisy_image = blurred_image + np.random.normal(0, noise_variance, blurred_image.shape)

# 函数执行维纳滤波
def wiener_deblur(image, kernel, noise_variance):
    # 计算核的功率谱
    kernel_fft = np.fft.fft2(kernel, image.shape)
    kernel_power_spectrum = np.abs(kernel_fft) ** 2

    # 计算图像的功率谱
    image_fft = np.fft.fft2(image)
    image_power_spectrum = np.abs(image_fft) ** 2

    # 执行维纳去模糊
    deblurred_image_fft = np.conj(kernel_fft) / (kernel_power_spectrum + noise_variance)
    deblurred_image_fft *= image_fft
    deblurred_image = np.fft.ifft2(deblurred_image_fft).real
    return deblurred_image

# 执行维纳去模糊
wiener_deblurred_image = wiener_deblur(noisy_image, kernel, noise_variance)

# 函数执行逆滤波
def direct_inverse_deblur(image, kernel):
    # 执行逆滤波
    deblurred_image = cv2.filter2D(image, -1, np.linalg.pinv(kernel))
    return deblurred_image

# 执行逆滤波
direct_inverse_deblurred_image = direct_inverse_deblur(noisy_image, kernel)

# 使用matplotlib显示结果
plt.figure(figsize=(12, 10))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为中文宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题

plt.subplot(221), plt.imshow(original_image, cmap='gray')
plt.title('原始图像'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(blurred_image, cmap='gray')
plt.title('退化图像'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(wiener_deblurred_image, cmap='gray')
plt.title('维纳滤波复原'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(direct_inverse_deblurred_image, cmap='gray')
plt.title('逆滤波复原'), plt.xticks([]), plt.yticks([])

plt.show()
