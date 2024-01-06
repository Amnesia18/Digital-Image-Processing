# Since we need to modify the provided code to work with the uploaded image,
# let's redefine the custom erosion and dilation functions, this time ensuring
# that they work with grayscale images. The uploaded image seems to be binary
# (black and white), so we will treat it as grayscale.

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the uploaded image
file_path = 'shiyan9.png'
image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale mode

# Define the erosion function
def custom_erosion(image, kernel_size, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = np.copy(image)
    
    # Add padding to the image to handle edges
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(eroded_image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
    
    for _ in range(iterations):
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                # Apply the kernel and assign the min value to the eroded image
                eroded_image[y, x] = np.min(padded_image[y:y+kernel_size, x:x+kernel_size])
    
    return eroded_image

# Define the dilation function
def custom_dilation(image, kernel_size, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = np.copy(image)
    
    # Add padding to the image to handle edges
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(dilated_image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
    
    for _ in range(iterations):
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                # Apply the kernel and assign the max value to the dilated image
                dilated_image[y, x] = np.max(padded_image[y:y+kernel_size, x:x+kernel_size])
    
    return dilated_image

# Define a 3x3 kernel
kernel_size = 5

# Perform erosion and dilation using the custom functions
eroded_image = custom_erosion(image, kernel_size)
dilated_image = custom_dilation(image, kernel_size)

# Define the opening operation function
def custom_opening(image, kernel_size):
    eroded = custom_erosion(image, kernel_size)
    opened = custom_dilation(eroded, kernel_size)
    return opened

# Define the closing operation function
def custom_closing(image, kernel_size):
    dilated = custom_dilation(image, kernel_size)
    closed = custom_erosion(dilated, kernel_size)
    return closed

# Apply the custom opening and closing operations
opened_image = custom_opening(image, kernel_size)
closed_image = custom_closing(image, kernel_size)

# Display the results
plt.figure(figsize=(10, 8))

titles = ['Original Image', 'Eroded Image', 'Dilated Image', 'Opened Image', 'Closed Image']
images = [image, eroded_image, dilated_image, opened_image, closed_image]

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
