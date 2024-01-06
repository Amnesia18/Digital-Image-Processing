import cv2

# 读取第一张图像
image1 = cv2.imread("rice.tif")

# 读取第二张图像
image2 = cv2.imread("c:\\Users\\Amnesia\\Desktop\\kb.png")

# 确保两张图像具有相同的大小
# 如果两张图像大小不同，可以调整它们的大小为相同的尺寸
if image1.shape != image2.shape:
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# 执行图像减法操作
result_image = cv2.subtract(image1, image2)

# 保存结果图像
cv2.imwrite("result_image.jpg", result_image)

print("图像减法完成")
