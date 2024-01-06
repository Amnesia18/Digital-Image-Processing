#-*- coding: utf-8 -*-

# 导入包
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image

#读取图片，并转为数组
im = np.array(Image.open("t.png").convert('L'))
print(im)
im_c = 255 - im

# 隐藏x轴和y轴
plt.axes().get_xaxis().set_visible(False)
plt.axes().get_yaxis().set_visible(False)

# 灰度显示
plt.gray()

# 显示图片
plt.imshow(im_c)

# #输出图中的最大和最小像素值
print(int(im_c.min()),int(im_c.max()))

# 显示图片
plt.show()
