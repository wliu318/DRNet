import os
import random
from shutil import copyfile

def split_dataset(input_images_dir1, input_images_dir2, input_labels_dir, output_dir1, output_dir2,output_labels_dir, split_ratio=(0.80, 0.002)):
    # 创建输出目录结构
    newdoc='train_5'
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(os.path.join(output_dir1, newdoc), exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    os.makedirs(os.path.join(output_dir2, newdoc), exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(os.path.join(output_labels_dir, newdoc), exist_ok=True)
    # os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    # os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    # os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    # os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    # os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_images_dir1) if f.endswith('.jpg')]
    num_images = len(image_files)

    # 随机打乱图片顺序
    random.shuffle(image_files)

    # 计算划分的数量
    num_train = int(num_images * split_ratio[0])
    # num_val = int(num_images * split_ratio[1])
    # num_test = num_images - num_train - num_val

    # 分割图片和标签文件
    for i, image_file in enumerate(image_files):
        if i < num_train:
            # 构建对应的标签文件名
            images_file2 = os.path.splitext(image_file)[0] + '.jpeg'
            label_file = os.path.splitext(image_file)[0] + '.txt'
            # 复制图片文件
            copyfile(os.path.join(input_images_dir1, image_file), os.path.join(output_dir1, newdoc, image_file))
            copyfile(os.path.join(input_images_dir2, images_file2), os.path.join(output_dir2, newdoc, images_file2))
           # 复制标签文件
            copyfile(os.path.join(input_labels_dir, label_file), os.path.join(output_labels_dir, newdoc, label_file))

# 调用划分函数
split_dataset('E:/deeplearning/dataset/FLIR-align-3class/visible/train', 'E:/deeplearning/dataset/FLIR-align-3class/infrared/train', 'E:/deeplearning/dataset/FLIR-align-3class/labels/train', 'E:/deeplearning/dataset/FLIR-align-3class/visible/', 'E:/deeplearning/dataset/FLIR-align-3class/infrared/', 'E:/deeplearning/dataset/FLIR-align-3class/labels/')

