import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def butterworth_bandpass_filter(source, center_x, center_y, bandwidth, order=1):
    height, width = source.shape
    center_x = center_x
    center_y = center_y

    # Create a meshgrid for the frequency domain
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x - center_x
    y = y - center_y

    # Calculate the distance from the center
    distance = np.sqrt(x**2 + y**2)

    # Create the butterworth bandpass filter
    filter = 1 / (1 + (distance / bandwidth)**(2 * order))

    # Apply the filter to the source spectrum
    filtered_spectrum = source * filter

    return filtered_spectrum

img_path = "len_grey.png"  # Replace with the path to your "Lena" image
src = np.array(Image.open(img_path).convert("L"))

# Perform Fourier transformation
fft_src = np.fft.fft2(src)
center_x, center_y = src.shape[1] // 2, src.shape[0] // 2  # Center coordinates
bandwidth = 50  # Adjust the bandwidth as needed
order = 2  # Adjust the order as needed

# Apply bandpass filter
filtered_spectrum = butterworth_bandpass_filter(fft_src, center_x, center_y, bandwidth, order)

# Perform inverse Fourier transformation
filtered_image = np.abs(np.fft.ifft2(filtered_spectrum)).astype(np.uint8)

# Display the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(src, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap="gray")
plt.title("Bandpass Filtered Image")
plt.axis("off")

plt.show()
