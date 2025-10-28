# -*- coding:utf-8 -*-
# move all the xml to one folder
# ------------kevin---------------

import os
import shutil
from tqdm import tqdm
from PIL import Image

src_filepath = "E:/deeplearning/dataset/VEDI/1024/raw/rgb"
files_1 = os.listdir(src_filepath)  # set

output_path = "E:/deeplearning/dataset/VEDI/1024/raw/yolo/rgb"

to_remove = "_co"

def convert_png_to_jpg(oldfile,newfile):
    # 获取文件的目录和名称（不带扩展名）
    directory, filename = os.path.split(oldfile)
    name, ext = os.path.splitext(filename)

    # 确保文件是 PNG 格式
    if ext.lower() != '.png':
        print(f"文件 {oldfile} 不是 PNG 格式，跳过转换。")
        return

    # 打开 PNG 文件
    with Image.open(oldfile) as img:
        # 转换为 RGB 模式（因为 JPG 不支持透明度）
        img_rgb = img.convert('RGB')
        img_rgb.save(newfile, 'JPEG')


for filename_1 in tqdm(files_1):
    src_path = os.path.join(src_filepath, filename_1)
    new_filename = filename_1.replace(to_remove, "")
    name, ext = os.path.splitext(new_filename)
    new_path = os.path.join(output_path, name + '.jpg')
    # 示例用法
    convert_png_to_jpg(src_path, new_path)

    # shutil.copy(src_path, new_path)



