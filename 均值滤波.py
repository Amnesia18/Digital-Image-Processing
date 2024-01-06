import cv2 as cv
import numpy as np


#均值滤波
def meansBlur(src, ksize):
    '''
    :param src: input image
    :param ksize:kernel size
    :return dst: output image
    '''
    dst = np.copy(src)  #创建输出图像
    kernel = np.ones((ksize, ksize))  # 卷积核
    padding_num = int((ksize - 1) / 2)  #需要补0
    dst = np.pad(dst, (padding_num, padding_num), mode="constant", constant_values=0)
    w, h = dst.shape
    dst = np.copy(dst)
    for i in range(padding_num, w - padding_num):
        for j in range(padding_num, h - padding_num):
            dst[i, j] = np.sum(kernel * dst[i - padding_num:i + padding_num + 1, j - padding_num:j + padding_num + 1]) \
                        // (ksize ** 2)
    dst = dst[padding_num:w - padding_num, padding_num:h - padding_num]  #把操作完多余的0去除，保证尺寸一样大
    return dst


img_path = r"sp.png"
img = cv.imread(img_path,0)
dst = meansBlur(img,7)
cv.imshow('src',img)
cv.imshow('dst',dst)
print(dst)
cv.waitKey(0)
