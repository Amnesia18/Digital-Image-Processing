import cv2
import numpy as np

def rle_encode(image):
    # 将图像转换为一维列表
    pixels = image.flatten()
    # 使用迭代器生成像素值和它们的计数
    from itertools import groupby
    rle = [(label, sum(1 for _ in group)) for label, group in groupby(pixels)]
    return rle

# 读取图像并转换为灰度
image_path = 'lena.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 应用行程压缩
rle_compressed = rle_encode(image)

# 计算压缩后的尺寸
compressed_size_rle = len(rle_compressed) * 2  # 每个行程需要两个值：像素值和计数
print(compressed_size_rle)

# 显示压缩后的图像（RLE压缩不改变可视图像）
cv2.imshow('RLE Compressed Image', image)
cv2.waitKey(0)

# 计算原图和压缩后的尺寸
original_size = image.shape[0] * image.shape[1]
print(original_size)
