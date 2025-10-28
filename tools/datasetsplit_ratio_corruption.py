import os
import random
from shutil import copyfile

def split_dataset(dataset_dir, train_test, corruptiondir,split_ratio=(0.05, 0.45, 0.45)):
    input_raw_rgb_dir = os.path.join(dataset_dir, 'visible', train_test)
    input_raw_ir_dir = os.path.join(dataset_dir, 'infrared', train_test)
    input_raw_labels_dir = os.path.join(dataset_dir, 'labels', train_test)
    input_quality_dir = os.path.join(dataset_dir, 'qualities', train_test)
    input_ircorruption_dir = os.path.join(dataset_dir, 'corruption/basic/ir zeros/infrared', train_test)
    input_rgbcorruption_dir = os.path.join(dataset_dir, 'corruption/basic/rgb zeros/visible', train_test)
    input_irqualitycorruption_dir = os.path.join(dataset_dir, 'corruption/basic/ir zeros/qualities', train_test)
    input_rgbqualitycorruption_dir = os.path.join(dataset_dir, 'corruption/basic/rgb zeros/qualities', train_test)


    out_ir_dir = os.path.join(dataset_dir, 'corruption/dual_modal', corruptiondir, 'infrared', train_test)
    os.makedirs(out_ir_dir, exist_ok=True)

    out_rgb_dir = os.path.join(dataset_dir, 'corruption/dual_modal', corruptiondir, 'visible', train_test)
    os.makedirs(out_rgb_dir, exist_ok=True)

    out_label_dir = os.path.join(dataset_dir, 'corruption/dual_modal', corruptiondir, 'labels', train_test)
    os.makedirs(out_label_dir, exist_ok=True)

    out_quality_dir = os.path.join(dataset_dir, 'corruption/dual_modal', corruptiondir, 'qualities', train_test)
    os.makedirs(out_quality_dir, exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_raw_rgb_dir) if f.endswith('.jpg')]
    num_images = len(image_files)

    # 随机打乱图片顺序
    random.shuffle(image_files)

    # 计算划分的数量
    num_normal = int(num_images * split_ratio[0])
    num_ir = int(num_images * split_ratio[1])
    num_rgb = int(num_images * split_ratio[2])

    # 分割图片和标签文件
    for i, image_file in enumerate(image_files):
        label_file = os.path.splitext(image_file)[0] + '.txt'
        quality_file = label_file
        copyfile(os.path.join(input_raw_labels_dir, label_file), os.path.join(out_label_dir, label_file))

        if i < num_normal:
            rgb_image=image_file
            copyfile(os.path.join(input_raw_rgb_dir, rgb_image), os.path.join(out_rgb_dir, rgb_image))

            ir_image= os.path.splitext(image_file)[0] + '.jpg'
            copyfile(os.path.join(input_raw_ir_dir, ir_image), os.path.join(out_ir_dir, ir_image))

            copyfile(os.path.join(input_quality_dir, quality_file), os.path.join(out_quality_dir, quality_file))

        elif i < num_normal+num_ir:
            rgb_image=image_file
            copyfile(os.path.join(input_raw_rgb_dir, rgb_image), os.path.join(out_rgb_dir, rgb_image))

            ir_image= os.path.splitext(image_file)[0] + '.jpg'
            copyfile(os.path.join(input_ircorruption_dir, ir_image), os.path.join(out_ir_dir, ir_image))

            copyfile(os.path.join(input_irqualitycorruption_dir, quality_file), os.path.join(out_quality_dir, quality_file))

        else:
            rgb_image=image_file
            copyfile(os.path.join(input_rgbcorruption_dir, rgb_image), os.path.join(out_rgb_dir, rgb_image))

            ir_image= os.path.splitext(image_file)[0] + '.jpg'
            copyfile(os.path.join(input_raw_ir_dir, ir_image), os.path.join(out_ir_dir, ir_image))

            copyfile(os.path.join(input_rgbqualitycorruption_dir, quality_file), os.path.join(out_quality_dir, quality_file))


# 调用划分函数
split_dataset(dataset_dir='E:/deeplearning/dataset/M3FD/M3FD_Detection', train_test='train', corruptiondir='ir_rgb_45', split_ratio=(0.05, 0.45, 0.45))
