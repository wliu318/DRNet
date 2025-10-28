import os
from PIL import Image
from tqdm import tqdm
# from shutil import copyfile

def is_txt_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() == '.txt'

source_path = 'E:/deeplearning/dataset/simd/yolo/train_100_airplane'  # 原始的标签路径

save_file_path = 'E:/deeplearning/dataset/simd/yolo/train_big_airplane'  # 修改后的标签路径
os.makedirs(save_file_path, exist_ok=True)

plane_maxwidth = 0
plane_maxheight = 0
plane_maxsize = plane_maxwidth * plane_maxheight

plane_minwidth = 10240
plane_minheight = 10240
plane_minsize = plane_minwidth * plane_minheight

labels_name = os.listdir(source_path)  # 获得每一个标签名字的列表 / os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
for name in tqdm(labels_name):   # 遍历每一个文件
    txtname = source_path + "/" + name  # 获取每一个文件的文件名
    if is_txt_file(txtname):
        image_file = os.path.splitext(name)[0] + '.jpg'
        with Image.open(os.path.join(source_path, image_file)) as img:
            width, height = img.size

        read_file = open(txtname, 'r')  # 读取txt_file_path/labels路径中的文件，r表示以只读方式打开文件
        fline = read_file.readlines()  # 读取txt文件中每一行 / readlines()表示读取整行 / fline是列表类型，fline列表里的元素是str类型

        for j in fline:  # 遍历txt文件中每一行
            data_line = j.split()
            if plane_maxsize < float(data_line[3]) * width * float(data_line[4]) * height:
                plane_maxsize = float(data_line[3]) * width * float(data_line[4]) * height
                plane_maxwidth = float(data_line[3]) * width
                plane_maxheight = float(data_line[4]) * height
            if plane_minsize > float(data_line[3]) * width * float(data_line[4]) * height:
                plane_minsize = float(data_line[3]) * width * float(data_line[4]) * height
                plane_minwidth = float(data_line[3]) * width
                plane_minheight = float(data_line[4]) * height
            # if data_line[0] == '0':
            #     save_txt = open(save_file_path + "/" + name, 'a+')  # 读取save_file_path/labels路径中的文件. w+表示打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。
            #     b = " ".join(data_line)  # 将列表转换成字符串类型，且用空格分割
            #     save_txt.write(b)  # 写入新的文件中
            #     save_txt.write('\n')  # 换行
            #
            #     image_file = os.path.splitext(name)[0] + '.jpg'
            #     copyfile(os.path.join(source_path, image_file), os.path.join(save_file_path, image_file))


print(f"The maximum size of the airplane is size:{plane_maxsize}, width:{plane_maxwidth}, height{plane_maxheight}.")
print(f"The minimum size of the airplane is size:{plane_minsize}, width:{plane_minwidth}, height{plane_minheight}.")