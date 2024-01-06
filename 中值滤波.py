#utf-8
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def MedianFilter(src, dst, k=7, padding=None):  # 进行中值滤波，选择的是忽略边缘的滤波方式
    imarray = np.array(Image.open(src).convert('L'))  # 也可以采用convert直接转换成灰度图
    height, width = imarray.shape

    if not padding:  # 当没有边缘时
        edge = int((k - 1) / 2)  # 边缘忽略值
        if height - 1 - edge <= edge or width - 1 - edge <= edge:
            print("The parameter k is to large.")
            return None
        new_arr = np.zeros((height, width), dtype="uint8")
        for i in range(height):
            for j in range(width):
                if i <= edge - 1 or i >= height - 1 - edge or j <= edge - 1 or j >= height - edge - 1:
                    new_arr[i, j] = imarray[i, j]
                else:  # 没有设计排序算法，直接使用Numpy中的寻找中值函数
                    new_arr[i, j] = np.median(imarray[i - edge:i + edge + 1, j - edge:j + edge + 1])
        new_im = Image.fromarray(new_arr)
        new_im.save(dst)


src = "sp.png"
dst = "lena_saltpepper_finish7.png"

MedianFilter(src, dst)

img1 = Image.open(src)
img2 = Image.open(dst)

# Create a figure and axis
fig, ax = plt.subplots(1, 2)

# Display the images
ax[0].imshow(img1, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(img2, cmap='gray')
ax[1].set_title('Filtered Image')
ax[1].axis('off')

# Show the figure
plt.show()

