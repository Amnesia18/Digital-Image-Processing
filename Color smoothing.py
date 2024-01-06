import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image = cv2.imread('digital signal processing/lena.tif')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1. 应用7x7均值滤波器
kernel_size = (7, 7)
image_blurred = cv2.blur(image, kernel_size)

# 2. 仅对红色分量应用7x7均值滤波器
R, G, B = cv2.split(image)
R_blurred = cv2.blur(R, kernel_size)
image_R_blurred = cv2.merge([R_blurred, G, B])

# 3. 对每个分量应用直方图均衡化
R_equalized = cv2.equalizeHist(R)
G_equalized = cv2.equalizeHist(G)
B_equalized = cv2.equalizeHist(B)
image_equalized = cv2.merge([R_equalized, G_equalized, B_equalized])
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)

    # 添加椒盐噪声
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords] = 1

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords] = 0

    return noisy_image
# 4. 添加椒盐噪声并应用中值滤波器
noisy_image = add_salt_and_pepper_noise(image, 0.05, 0.05)
median_filtered = cv2.medianBlur((noisy_image * 255).astype(np.uint8), 5)
def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.01
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = image + gauss
    return np.clip(noisy_image, 0, 1)
# 5. 添加高斯噪声并应用高斯滤波器
gaussian_noisy_image = add_gaussian_noise(image)
gaussian_filtered = cv2.GaussianBlur((gaussian_noisy_image * 255).astype(np.uint8), (5, 5), 0)

# 显示所有结果
plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(image_blurred)
plt.title('7x7 Mean Filtered')
plt.axis('off')

plt.subplot(3, 2, 3)
plt.imshow(image_R_blurred)
plt.title('Red Component 7x7 Mean Filtered')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(image_equalized)
plt.title('Histogram Equalized')
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(median_filtered)
plt.title('Median Filter on Salt and Pepper Noise')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(gaussian_filtered)
plt.title('Gaussian Filter on Gaussian Noise')
plt.axis('off')

plt.tight_layout()
plt.show()
