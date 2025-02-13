from PIL import Image
import numpy as np

# 读取图片
image_path = 'E:\\xingyou_ubuntu_log_2025_0201\AVAS_autonomous_vehicle_auto_shipment\yolo-series\\bdd100k\labels\drivable\masks\\train\\0000f77c-62c2a288.png'  # 替换为你的图片路径
img = Image.open(image_path)

# 将图片转换为 NumPy 数组
img_array = np.array(img)

# 找到图片中的最大像素值
max_value = np.max(img_array)

print(f"图片中的最高像素值是: {max_value}")