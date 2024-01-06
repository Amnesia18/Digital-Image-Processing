# import cv2
# import numpy as np

# # 读取灰度图像
# image = cv2.imread('digital signal processing/finger.png', cv2.IMREAD_GRAYSCALE)

# # 设定阈值（这里使用127作为阈值，你可以根据需要自行调整）
# threshold_value = 127

# # 使用cv2.threshold()函数进行全局阈值处理
# ret, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# # 显示原始图像和阈值处理后的图像
# cv2.imshow('Original Image', image)
# cv2.imshow('Thresholded Image', thresholded_image)

# # 等待按键并关闭窗口
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # 读取灰度图像
# image = cv2.imread('digital signal processing/coins.png', cv2.IMREAD_GRAYSCALE)

# # 使用Otsu阈值分割
# ret, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # 显示原始图像和Otsu阈值分割后的图像
# cv2.imshow('Original Image', image)
# cv2.imshow('Otsu Thresholded Image', thresholded_image)

# # 等待按键并关闭窗口
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # 读取灰度图像
# image = cv2.imread('digital signal processing/bgztp.png', cv2.IMREAD_GRAYSCALE)

# # 定义分块大小（这里使用10x10的小块，你可以根据需要调整）
# block_size = 10

# # 获取图像的高度和宽度
# height, width = image.shape

# # 创建一个与图像相同大小的空白图像，用于存储分块可变阈值分割结果
# result_image = np.zeros((height, width), dtype=np.uint8)

# # 分块可变阈值分割
# for y in range(0, height, block_size):
#     for x in range(0, width, block_size):
#         # 获取当前分块
#         block = image[y:y+block_size, x:x+block_size]

#         # 计算当前分块的阈值
#         threshold_value = cv2.mean(block)[0]

#         # 使用阈值对当前分块进行二值化
#         ret, thresholded_block = cv2.threshold(block, threshold_value, 255, cv2.THRESH_BINARY)

#         # 将二值化的分块放回结果图像中
#         result_image[y:y+block_size, x:x+block_size] = thresholded_block

# # 显示原始图像和分块可变阈值分割后的图像
# cv2.imshow('Original Image', image)
# cv2.imshow('Block-wise Variable Thresholding', result_image)

# # 等待按键并关闭窗口
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

def my_Otsu(im):
    # Apply Gaussian blur to the subregion
    blur = cv2.GaussianBlur(im, (5, 5), 0)
    # Use OpenCV's Otsu's thresholding
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def divide_image(im, h_divisions, w_divisions):
    h, w = im.shape[:2]
    h_sub = h // h_divisions
    w_sub = w // w_divisions
    
    # Adjust the last subregion size if the image isn't evenly divisible
    subregions = []
    for i in range(h_divisions):
        for j in range(w_divisions):
            i_start = i * h_sub
            j_start = j * w_sub
            i_end = h if i == h_divisions - 1 else (i + 1) * h_sub
            j_end = w if j == w_divisions - 1 else (j + 1) * w_sub
            subregions.append(im[i_start:i_end, j_start:j_end])
    return subregions

def reconstruct_image(subregions, h_divisions, w_divisions):
    # Concatenate subregions back into an image
    rows = [np.hstack(subregions[i*w_divisions:(i+1)*w_divisions]) for i in range(h_divisions)]
    imd = np.vstack(rows)
    return imd

# Replace 'path_to_image' with the actual image path
im = cv2.imread('digital signal processing/bgztp.png', 0)

# Parameters for dividing the image into subregions
h_divisions = 2
w_divisions = 3

# Divide the image and apply Otsu's thresholding to each subregion
subregions = divide_image(im, h_divisions, w_divisions)
thresh_subregions = [my_Otsu(sr) for sr in subregions]

# Reconstruct the divided image from the thresholded subregions
im_segmented = reconstruct_image(thresh_subregions, h_divisions, w_divisions)

# Save and display the result
cv2.imwrite('result_segmented.jpg', im_segmented)
cv2.imshow('Segmented Image', im_segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()
