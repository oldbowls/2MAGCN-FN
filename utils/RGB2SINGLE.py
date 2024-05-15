import cv2
import numpy as np

# 读取彩色图像
input_image = cv2.imread(r"C:\Users\11761\Desktop\1.png")

# 转换为灰度图像
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# 使用Canny边缘检测算法
edges = cv2.Canny(gray_image, 50, 150)  # 调整阈值以获取合适的边缘

# 将边缘图转换为彩色图像
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# 将彩色图像和边缘图像叠加
output_image = cv2.addWeighted(input_image, 0.7, edges_colored, 0.3, 0)

# 保存结果
cv2.imwrite('output_lineart.jpg', output_image)

# 显示结果
cv2.imshow('Original Image', input_image)
cv2.imshow('Lineart Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
