import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. 生成黑白块测试图像
def generate_checkerboard(size):
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, 20):
        for j in range(0, size, 20):
            img[i:i+10, j:j+10] = 255
    return img

# 2. 添加高斯噪声并应用均值滤波和中值滤波
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_image = cv2.add(image, noise)
    return noisy_image

# 3. 添加椒盐噪声
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = image.copy()
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

# 4. 应用合适的滤波方法去噪
def denoise_image(image, method):
    if method == 'mean':
        denoised_image = cv2.blur(image, (5, 5))
    elif method == 'median':
        denoised_image = cv2.medianBlur(image, 5)
    else:
        raise ValueError("Invalid denoising method")
    return denoised_image

# 生成测试图像
size = 256
checkerboard_image = generate_checkerboard(size)

# 添加高斯噪声并进行去噪
gaussian_noisy_image = add_gaussian_noise(checkerboard_image)
gaussian_denoised_mean = denoise_image(gaussian_noisy_image, 'mean')
gaussian_denoised_median = denoise_image(gaussian_noisy_image, 'median')

# 添加椒盐噪声并进行去噪
salt_pepper_noisy_image = add_salt_and_pepper_noise(checkerboard_image, salt_prob=0.01, pepper_prob=0.01)
sp_denoised_mean = denoise_image(salt_pepper_noisy_image, 'mean')
sp_denoised_median = denoise_image(salt_pepper_noisy_image, 'median')

# 叠加高斯噪声和椒盐噪声并进行去噪
combined_noisy_image = add_gaussian_noise(salt_pepper_noisy_image)
combined_denoised_mean = denoise_image(combined_noisy_image, 'mean')
combined_denoised_median = denoise_image(combined_noisy_image, 'median')

plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示结果
plt.figure(figsize=(12, 10))

plt.subplot(331)
plt.title('Checkerboard')
plt.imshow(checkerboard_image, cmap='gray')

plt.subplot(332)
plt.title('Gaussian Noisy Image')
plt.imshow(gaussian_noisy_image, cmap='gray')

plt.subplot(333)
plt.title('Gaussian Denoised (Mean)')
plt.imshow(gaussian_denoised_mean, cmap='gray')

plt.subplot(334)
plt.title('Salt & Pepper Noisy Image')
plt.imshow(salt_pepper_noisy_image, cmap='gray')

plt.subplot(335)
plt.title('SP Denoised (Mean)')
plt.imshow(sp_denoised_mean, cmap='gray')

plt.subplot(336)
plt.title('Combined Noisy Image')
plt.imshow(combined_noisy_image, cmap='gray')

plt.subplot(337)
plt.title('Combined Denoised (Mean)')
plt.imshow(combined_denoised_mean, cmap='gray')

plt.subplot(338)
plt.title('Gaussian Denoised (Median)')
plt.imshow(gaussian_denoised_median, cmap='gray')

plt.subplot(339)
plt.title('SP Denoised (Median)')
plt.imshow(sp_denoised_median, cmap='gray')

plt.tight_layout()
plt.show()
