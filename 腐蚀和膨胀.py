# Define the custom morphology functions without using OpenCV
import cv2
import numpy as np
from matplotlib import pyplot as plt

def custom_erosion(image, kernel):
    # Create padding around the image
    padded_image = np.pad(image, ((1, 1), (1, 1)), 'constant')
    eroded_image = np.zeros_like(image)

    for y in range(1, padded_image.shape[0] - 1):
        for x in range(1, padded_image.shape[1] - 1):
            # Erosion takes the minimum value of the kernel area
            eroded_image[y-1, x-1] = np.min(padded_image[y-1:y+2, x-1:x+2][kernel==1])
    return eroded_image

def custom_dilation(image, kernel):
    # Create padding around the image
    padded_image = np.pad(image, ((1, 1), (1, 1)), 'constant')
    dilated_image = np.zeros_like(image)

    for y in range(1, padded_image.shape[0] - 1):
        for x in range(1, padded_image.shape[1] - 1):
            # Dilation takes the maximum value of the kernel area
            dilated_image[y-1, x-1] = np.max(padded_image[y-1:y+2, x-1:x+2][kernel==1])
    return dilated_image

def custom_opening(image, kernel):
    return custom_dilation(custom_erosion(image, kernel), kernel)

def custom_closing(image, kernel):
    return custom_erosion(custom_dilation(image, kernel), kernel)

# Define a 3x3 kernel
kernel = np.ones((3, 3), dtype=np.uint8)

# Load the uploaded fingerprint image
fingerprint_image = cv2.imread('finger.png', cv2.IMREAD_GRAYSCALE)

# Apply the morphology operations
dilated = custom_dilation(fingerprint_image, kernel)
opened = custom_opening(dilated, kernel)
eroded = custom_erosion(opened, kernel)
closed = custom_closing(eroded, kernel)
final = custom_dilation(closed, kernel)

# Display the results
plt.figure(figsize=(12, 8))
titles = ['Original', 'Dilated', 'Opened', 'Eroded', 'Closed', 'Final']
images = [fingerprint_image, dilated, opened, eroded, closed, final]

for i in range(len(images)):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
