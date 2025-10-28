import os
import random
from shutil import copyfile

def split_dataset(dataset, in_subdir, out_subdir, split_ratio=(0.975, 0.025)):
    # 创建输出目录结构
    # split_ratio=(0.975, 0.025):8:2

    input_images_dir1 = os.path.join(dataset, 'visible', in_subdir)
    input_images_dir2 = os.path.join(dataset, 'infrared', in_subdir)
    input_labels_dir  = os.path.join(dataset, 'labels', in_subdir)

    output_dir1       = os.path.join(dataset, 'visible', out_subdir)
    os.makedirs(output_dir1, exist_ok=True)
    output_dir2       = os.path.join(dataset, 'infrared', out_subdir)
    os.makedirs(output_dir2, exist_ok=True)
    output_labels_dir = os.path.join(dataset, 'labels', out_subdir)
    os.makedirs(output_labels_dir, exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_images_dir1) if f.endswith('.jpg')]
    num_images = len(image_files)

    # 随机打乱图片顺序
    random.shuffle(image_files)

    # 计算划分的数量
    num_train = int(num_images * split_ratio[0])
    num_test = int(num_images * split_ratio[1])


    # 分割图片和标签文件
    for i, image_file in enumerate(image_files):
        if i < num_train:
           images_file2 = os.path.splitext(image_file)[0] + '.jpeg'
           label_file = os.path.splitext(image_file)[0] + '.txt'
           copyfile(os.path.join(input_images_dir1, image_file), os.path.join(output_dir1, image_file))
           copyfile(os.path.join(input_images_dir2, images_file2), os.path.join(output_dir2, images_file2))
           copyfile(os.path.join(input_labels_dir, label_file), os.path.join(output_labels_dir, label_file))

        elif i < num_train + num_test:
           images_file2 = os.path.splitext(image_file)[0] + '.jpeg'
           label_file = os.path.splitext(image_file)[0] + '.txt'

           mixfileext = 'ir blur'
           out_image1 = mixfileext + '_' + image_file
           out_image2 = mixfileext + '_' + images_file2
           out_labels = mixfileext + '_' + label_file
           in_images1 = os.path.join(dataset, 'corruption', mixfileext, 'visible', in_subdir)
           in_images2 = os.path.join(dataset, 'corruption', mixfileext, 'infrared', in_subdir)
           in_labels_dir = os.path.join(dataset, 'corruption', mixfileext, 'labels', in_subdir)
           copyfile(os.path.join(in_images1, image_file), os.path.join(output_dir1, out_image1))
           copyfile(os.path.join(in_images2, images_file2), os.path.join(output_dir2, out_image2))
           # 复制标签文件
           copyfile(os.path.join(in_labels_dir, label_file), os.path.join(output_labels_dir, out_labels))

           mixfileext = 'ir dropout'
           out_image1 = mixfileext + '_' + image_file
           out_image2 = mixfileext + '_' + images_file2
           out_labels = mixfileext + '_' + label_file
           in_images1 = os.path.join(dataset, 'corruption', mixfileext, 'visible', in_subdir)
           in_images2 = os.path.join(dataset, 'corruption', mixfileext, 'infrared', in_subdir)
           in_labels_dir = os.path.join(dataset, 'corruption', mixfileext, 'labels', in_subdir)
           copyfile(os.path.join(in_images1, image_file), os.path.join(output_dir1, out_image1))
           copyfile(os.path.join(in_images2, images_file2), os.path.join(output_dir2, out_image2))
           copyfile(os.path.join(in_labels_dir, label_file), os.path.join(output_labels_dir, out_labels))

           mixfileext = 'ir noise'
           out_image1 = mixfileext + '_' + image_file
           out_image2 = mixfileext + '_' + images_file2
           out_labels = mixfileext + '_' + label_file
           in_images1 = os.path.join(dataset, 'corruption', mixfileext, 'visible', in_subdir)
           in_images2 = os.path.join(dataset, 'corruption', mixfileext, 'infrared', in_subdir)
           in_labels_dir = os.path.join(dataset, 'corruption', mixfileext, 'labels', in_subdir)
           copyfile(os.path.join(in_images1, image_file), os.path.join(output_dir1, out_image1))
           copyfile(os.path.join(in_images2, images_file2), os.path.join(output_dir2, out_image2))
           copyfile(os.path.join(in_labels_dir, label_file), os.path.join(output_labels_dir, out_labels))

           mixfileext = 'ir zeros'
           out_image1 = mixfileext + '_' + image_file
           out_image2 = mixfileext + '_' + images_file2
           out_labels = mixfileext + '_' + label_file
           in_images1 = os.path.join(dataset, 'corruption', mixfileext, 'visible', in_subdir)
           in_images2 = os.path.join(dataset, 'corruption', mixfileext, 'infrared', in_subdir)
           in_labels_dir = os.path.join(dataset, 'corruption', mixfileext, 'labels', in_subdir)
           copyfile(os.path.join(in_images1, image_file), os.path.join(output_dir1, out_image1))
           copyfile(os.path.join(in_images2, images_file2), os.path.join(output_dir2, out_image2))
           copyfile(os.path.join(in_labels_dir, label_file), os.path.join(output_labels_dir, out_labels))

           mixfileext = 'RGB blur'
           out_image1 = mixfileext + '_' + image_file
           out_image2 = mixfileext + '_' + images_file2
           out_labels = mixfileext + '_' + label_file
           in_images1 = os.path.join(dataset, 'corruption', mixfileext, 'visible', in_subdir)
           in_images2 = os.path.join(dataset, 'corruption', mixfileext, 'infrared', in_subdir)
           in_labels_dir = os.path.join(dataset, 'corruption', mixfileext, 'labels', in_subdir)
           copyfile(os.path.join(in_images1, image_file), os.path.join(output_dir1, out_image1))
           copyfile(os.path.join(in_images2, images_file2), os.path.join(output_dir2, out_image2))
           copyfile(os.path.join(in_labels_dir, label_file), os.path.join(output_labels_dir, out_labels))

           mixfileext = 'RGB brightness'
           out_image1 = mixfileext + '_' + image_file
           out_image2 = mixfileext + '_' + images_file2
           out_labels = mixfileext + '_' + label_file
           in_images1 = os.path.join(dataset, 'corruption', mixfileext, 'visible', in_subdir)
           in_images2 = os.path.join(dataset, 'corruption', mixfileext, 'infrared', in_subdir)
           in_labels_dir = os.path.join(dataset, 'corruption', mixfileext, 'labels', in_subdir)
           copyfile(os.path.join(in_images1, image_file), os.path.join(output_dir1, out_image1))
           copyfile(os.path.join(in_images2, images_file2), os.path.join(output_dir2, out_image2))
           copyfile(os.path.join(in_labels_dir, label_file), os.path.join(output_labels_dir, out_labels))

           mixfileext = 'RGB color cast'
           out_image1 = mixfileext + '_' + image_file
           out_image2 = mixfileext + '_' + images_file2
           out_labels = mixfileext + '_' + label_file
           in_images1 = os.path.join(dataset, 'corruption', mixfileext, 'visible', in_subdir)
           in_images2 = os.path.join(dataset, 'corruption', mixfileext, 'infrared', in_subdir)
           in_labels_dir = os.path.join(dataset, 'corruption', mixfileext, 'labels', in_subdir)
           copyfile(os.path.join(in_images1, image_file), os.path.join(output_dir1, out_image1))
           copyfile(os.path.join(in_images2, images_file2), os.path.join(output_dir2, out_image2))
           copyfile(os.path.join(in_labels_dir, label_file), os.path.join(output_labels_dir, out_labels))

           mixfileext = 'RGB colortemperature'
           out_image1 = mixfileext + '_' + image_file
           out_image2 = mixfileext + '_' + images_file2
           out_labels = mixfileext + '_' + label_file
           in_images1 = os.path.join(dataset, 'corruption', mixfileext, 'visible', in_subdir)
           in_images2 = os.path.join(dataset, 'corruption', mixfileext, 'infrared', in_subdir)
           in_labels_dir = os.path.join(dataset, 'corruption', mixfileext, 'labels', in_subdir)
           copyfile(os.path.join(in_images1, image_file), os.path.join(output_dir1, out_image1))
           copyfile(os.path.join(in_images2, images_file2), os.path.join(output_dir2, out_image2))
           copyfile(os.path.join(in_labels_dir, label_file), os.path.join(output_labels_dir, out_labels))

           mixfileext = 'RGB noise'
           out_image1 = mixfileext + '_' + image_file
           out_image2 = mixfileext + '_' + images_file2
           out_labels = mixfileext + '_' + label_file
           in_images1 = os.path.join(dataset, 'corruption', mixfileext, 'visible', in_subdir)
           in_images2 = os.path.join(dataset, 'corruption', mixfileext, 'infrared', in_subdir)
           in_labels_dir = os.path.join(dataset, 'corruption', mixfileext, 'labels', in_subdir)
           copyfile(os.path.join(in_images1, image_file), os.path.join(output_dir1, out_image1))
           copyfile(os.path.join(in_images2, images_file2), os.path.join(output_dir2, out_image2))
           copyfile(os.path.join(in_labels_dir, label_file), os.path.join(output_labels_dir, out_labels))

           mixfileext = 'RGB zeros'
           out_image1 = mixfileext + '_' + image_file
           out_image2 = mixfileext + '_' + images_file2
           out_labels = mixfileext + '_' + label_file
           in_images1 = os.path.join(dataset, 'corruption', mixfileext, 'visible', in_subdir)
           in_images2 = os.path.join(dataset, 'corruption', mixfileext, 'infrared', in_subdir)
           in_labels_dir = os.path.join(dataset, 'corruption', mixfileext, 'labels', in_subdir)
           copyfile(os.path.join(in_images1, image_file), os.path.join(output_dir1, out_image1))
           copyfile(os.path.join(in_images2, images_file2), os.path.join(output_dir2, out_image2))
           copyfile(os.path.join(in_labels_dir, label_file), os.path.join(output_labels_dir, out_labels))

# 调用划分函数
split_dataset('E:\\deeplearning\\dataset\\FLIR-align-3class', 'test', 'mix_test')

