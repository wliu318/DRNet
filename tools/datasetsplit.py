import os
import random
from shutil import copyfile

def split_dataset(input_dir, output_dir, split_ratio=(0.7, 0.3)):
    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    num_images = len(image_files)
    # 计算划分的数量
    num_train = int(num_images * split_ratio[0])
    num_test = num_images - num_train

    # 随机打乱图片顺序
    random.shuffle(image_files)

    # 分割图片和标签文件
    for i, image_file in enumerate(image_files):
        if i < num_train:
            set_name = 'train'
        # elif i < num_train + num_test:
        #     set_name = 'test'
        else:
            set_name = 'test'

        # 复制图片文件
        copyfile(os.path.join(input_dir, image_file), os.path.join(output_dir, set_name, image_file))

        # 构建对应的标签文件名
        label_file = os.path.splitext(image_file)[0] + '.txt'
        copyfile(os.path.join(input_dir, label_file), os.path.join(output_dir, set_name, label_file))

# 调用划分函数
split_dataset('E:/deeplearning/dataset/simd/yolo_airplane/raw/', 'E:/deeplearning/dataset/simd/yolo_airplane')

