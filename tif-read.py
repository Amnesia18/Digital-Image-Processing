import cv2

# 读取TIFF图像
tif_img = cv2.imread("2.tif")

# 转换为真彩色图像
true_color_img = tif_img

# 转换为灰度图像
gray_img = cv2.cvtColor(tif_img, cv2.COLOR_BGR2GRAY)

# 使用阈值方法将灰度图像转换为二值图像
_, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

# 显示图像
cv2.imshow("True Color Image", true_color_img)
cv2.imshow("Gray Image", gray_img)
cv2.imshow("Binary Image", binary_img)

# 等待用户按下任意键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
