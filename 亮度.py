from skimage import exposure, img_as_float, io
import matplotlib.pyplot as plt

# Load an image from a file (replace 'your_image.jpg' with the actual file path)
image = img_as_float(io.imread('1.bmp'))

gam1 = exposure.adjust_gamma(image, 2)    # Darken
gam2 = exposure.adjust_gamma(image, 0.5)  # Brighten

plt.figure('0.5和2的gamma对比图', figsize=(8, 8))

plt.subplot(131)
plt.title('Original Image')
plt.imshow(image, plt.cm.gray)
plt.axis('off')

plt.subplot(132)
plt.title('Gamma = 2')
plt.imshow(gam1, plt.cm.gray)
plt.axis('off')

plt.subplot(133)
plt.title('Gamma = 0.5')
plt.imshow(gam2, plt.cm.gray)
plt.axis('off')

plt.show()
