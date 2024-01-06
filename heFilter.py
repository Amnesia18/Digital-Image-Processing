import cv2
import numpy as np
import matplotlib.pyplot as plt

def homomorphic_filter(img, cutoff_freq, order, low_gain, high_gain):
    img_log = np.log1p(np.float64(img))
    img_fft = np.fft.fft2(img_log)
    rows, cols = img_fft.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float64)
    mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 1
    img_fft_filtered = img_fft * mask
    img_filtered = np.fft.ifft2(img_fft_filtered)
    img_filtered = np.real(img_filtered)
    img_filtered = np.expm1(img_filtered)
    img_filtered = (img_filtered - np.min(img_filtered)) / (np.max(img_filtered) - np.min(img_filtered))
    img_filtered = 255 * img_filtered
    img_filtered_low = low_gain * img_filtered
    img_filtered_high = high_gain * (img - img_filtered_low)
    return img_filtered_low + img_filtered_high

img = cv2.imread('len_grey.png', cv2.IMREAD_GRAYSCALE)
cutoff_freq = 30
order = 2
low_gain = 1.5
high_gain = 1.2

filtered_img = homomorphic_filter(img, cutoff_freq, order, low_gain, high_gain)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_img, cmap='gray')
plt.title('Homomorphic Filtered Image')
plt.axis('off')

plt.show()
