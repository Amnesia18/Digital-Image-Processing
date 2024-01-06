
import numpy as np
from scipy.fft import fft, ifft,fftshift

# 创建输入向量
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# 计算傅里叶变换
X = fft(x)

# 计算傅里叶反变换
x_ifft = ifft(X)

# 打印原始向量和反变换
print("原始向量 ", x)
print("傅里叶反变换", x_ifft.real)
#使用x乘以(-1)^n
n = len(x)
shifted_signal = x * (-1) ** np.arange(n)
X_shifted1 = fft(shifted_signal)
print("平移后向量1",X_shifted1)
#使用fftshif函数实现
X_shifted2 = fftshift(X)
print("平移后向量2",X_shifted2)