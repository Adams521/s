import os
from PIL import Image

# 假设你的.tif文件都在这个目录下
source_dir = 'Train_Data/Similar_Data'
# 如果你想把转换后的.png文件放在不同的地方，可以修改这个目录
target_root_dir = 'split_data'

# 确保目标根目录存在
os.makedirs(target_root_dir, exist_ok=True)

# 遍历目录中的所有文件
for filename in os.listdir(source_dir):
    if filename.endswith('.tif'):
        # 提取x的值
        x_value = filename.split('_')[0][3:]
        # 创建目标目录
        target_dir = os.path.join(target_root_dir, f"Sim{x_value}")
        os.makedirs(target_dir, exist_ok=True)
        
        # 打开.tif文件并转换为.png
        with Image.open(os.path.join(source_dir, filename)) as img:
            target_file = os.path.splitext(filename)[0] + '.png'
            img.save(os.path.join(target_dir, target_file))