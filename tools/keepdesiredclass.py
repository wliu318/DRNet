import os
from tqdm import tqdm

def is_txt_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() == '.txt'

txt_file_path = 'E:/deeplearning/dataset/simd/yolo/test'  # 原始的标签路径
save_file_path = 'E:/deeplearning/dataset/simd/yolo/test_airplane'  # 修改后的标签路径

labels_name = os.listdir(txt_file_path)  # 获得每一个标签名字的列表 / os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
files = os.listdir(save_file_path)

for name in tqdm(labels_name):   # 遍历每一个文件
    txtname = txt_file_path + "/" + name  # 获取每一个文件的文件名
    if is_txt_file(txtname):
        read_file = open(txtname, 'r')  # 读取txt_file_path/labels路径中的文件，r表示以只读方式打开文件
        fline = read_file.readlines()  # 读取txt文件中每一行 / readlines()表示读取整行 / fline是列表类型，fline列表里的元素是str类型
        save_txt = open(save_file_path + "/" + name, 'w+')  # 读取save_file_path/labels路径中的文件. w+表示打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。

        for j in fline:  # 遍历txt文件中每一行
            old_line = j.split()
            new_line = old_line

            # 删除类别
            # if old_line[0] == '0':
            #     new_line[0] = '1'
            # elif old_line[0] == '1':
            #     new_line[0] = '2'
            # elif old_line[0] == '2':
            #     new_line[0] = '3'
            # elif old_line[0] == '3':
            #     new_line[0] = '4'
            # elif old_line[0] == '4':
            #     new_line[0] = '5'
            # elif old_line[0] == '5':
            #     new_line[0] = '0'

            if old_line[0] == '0':
                b = " ".join(new_line)  # 将列表转换成字符串类型，且用空格分割
                save_txt.write(b)  # 写入新的文件中
                save_txt.write('\n')  # 换行