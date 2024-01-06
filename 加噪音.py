import cv2
import numpy as np
import os
import random
from matplotlib import pyplot as plt
from PIL import Image
def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
def process_images(folder_path):
    # 创建输出目录
    output_folder = os.path.join(folder_path, 'noisy_images')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 读取图像文件
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            
            # 添加椒盐噪声到彩色图像
            sp_noisy_img = sp_noise(img, prob=0.02)
            sp_img_path = os.path.join(output_folder, 'sp_' + img_name)
            cv2.imwrite(sp_img_path, sp_noisy_img)
            
            # 添加高斯噪声到彩色图像
            gauss_noisy_img = gasuss_noise(img, mean=0, var=0.009)
            gauss_img_path = os.path.join(output_folder, 'gauss_' + img_name)
            cv2.imwrite(gauss_img_path, gauss_noisy_img)

# 调用函数处理指定文件夹下的图像
folder_path = 'E:\\test\CapturedImages\CapturedImages'  # 请替换为您的图像目录路径
process_images(folder_path)
