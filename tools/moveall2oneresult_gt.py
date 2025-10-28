# -*- coding:utf-8 -*-
# move all the xml to one folder
# ------------kevin---------------

import os
import shutil
from tqdm import tqdm

dataset = "llvip"
method  = "ours"

subdir1 = dataset + "_ir_rgb_45_00"
subdir2 = dataset + "_ir_rgb_45_day_rgblost"
subdir3 = dataset + "_ir_rgb_45_night_irlost"

newfile  = os.path.join("E:\\deeplearning\\paper3\\paper method\\22.MADRM\\evaluation_script\\state_of_arts", dataset, method, subdir1, method + "_" + dataset + "_result.txt")
filepath = os.path.join("E:\\deeplearning\\paper3\\paper method\\22.MADRM\\evaluation_script\\state_of_arts", dataset, method, subdir1, "labels")

files_1 = os.listdir(filepath)  # set
imageid = 1
newdata = []
width = 640.0
height = 512.0
with open(newfile, 'w', encoding='utf-8') as f_out:
    for filename_1 in tqdm(files_1):
        tmp_path_1 = os.path.join(filepath, filename_1)
        with open(tmp_path_1, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:

                if method in ["ours", "icaf"]:
                    data = [float(ll) for ll in line.split(',')]
                    if data[5] >= 0.25:
                        if method == "ours":
                            tmpdata = str(imageid) + "," + \
                                      str(data[1]) + "," + \
                                      str(data[2]) + "," + \
                                      str(data[3]) + "," + \
                                      str(data[4]) + "," + \
                                      str(1)
                        else:
                            tmpdata = str(imageid) + "," + \
                                      str(data[1] - data[3] / 2.0) + "," + \
                                      str(data[2] - data[4] / 2.0) + "," + \
                                      str(data[3]) + "," + \
                                      str(data[4]) + "," + \
                                      str(1)
                        f_out.write(tmpdata + "\n")

                else:
                    data = [float(ll) for ll in line.split(' ')]
                    if data[5] >= 0.25:
                        tmpdata = str(imageid) + "," + \
                                  str((data[1] - data[3] / 2.0) * width) + "," + \
                                  str((data[2] - data[4] / 2.0) * height) + "," + \
                                  str(data[3] * width) + "," + \
                                  str(data[4] * height) + "," + \
                                  str(1)
                        f_out.write(tmpdata + "\n")

        imageid += 1  # label id+1

newfile  = os.path.join("E:\\deeplearning\\paper3\\paper method\\22.MADRM\\evaluation_script\\state_of_arts", dataset, method, subdir2, method + "_" + dataset + "_result.txt")
filepath = os.path.join("E:\\deeplearning\\paper3\\paper method\\22.MADRM\\evaluation_script\\state_of_arts", dataset, method, subdir2, "labels")

files_1 = os.listdir(filepath)  # set
imageid = 1
newdata = []
width = 640.0
height = 512.0
with open(newfile, 'w', encoding='utf-8') as f_out:
    for filename_1 in tqdm(files_1):
        tmp_path_1 = os.path.join(filepath, filename_1)
        with open(tmp_path_1, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:

                if method in ["ours", "icaf"]:
                    data = [float(ll) for ll in line.split(',')]
                    if data[5] >= 0.25:
                        if method == "ours":
                            tmpdata = str(imageid) + "," + \
                                      str(data[1]) + "," + \
                                      str(data[2]) + "," + \
                                      str(data[3]) + "," + \
                                      str(data[4]) + "," + \
                                      str(1)
                        else:
                            tmpdata = str(imageid) + "," + \
                                      str(data[1] - data[3] / 2.0) + "," + \
                                      str(data[2] - data[4] / 2.0) + "," + \
                                      str(data[3]) + "," + \
                                      str(data[4]) + "," + \
                                      str(1)
                        f_out.write(tmpdata + "\n")

                else:
                    data = [float(ll) for ll in line.split(' ')]
                    if data[5] >= 0.25:
                        tmpdata = str(imageid) + "," + \
                                  str((data[1] - data[3] / 2.0) * width) + "," + \
                                  str((data[2] - data[4] / 2.0) * height) + "," + \
                                  str(data[3] * width) + "," + \
                                  str(data[4] * height) + "," + \
                                  str(1)
                        f_out.write(tmpdata + "\n")

        imageid += 1  # label id+1


newfile  = os.path.join("E:\\deeplearning\\paper3\\paper method\\22.MADRM\\evaluation_script\\state_of_arts", dataset, method, subdir3, method + "_" + dataset + "_result.txt")
filepath = os.path.join("E:\\deeplearning\\paper3\\paper method\\22.MADRM\\evaluation_script\\state_of_arts", dataset, method, subdir3, "labels")

files_1 = os.listdir(filepath)  # set
imageid = 1
newdata = []
width = 640.0
height = 512.0
with open(newfile, 'w', encoding='utf-8') as f_out:
    for filename_1 in tqdm(files_1):
        tmp_path_1 = os.path.join(filepath, filename_1)
        with open(tmp_path_1, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:

                if method in ["ours", "icaf"]:
                    data = [float(ll) for ll in line.split(',')]
                    if data[5] >= 0.25:
                        if method == "ours":
                            tmpdata = str(imageid) + "," + \
                                      str(data[1]/2) + "," + \
                                      str(data[2]/2) + "," + \
                                      str(data[3]/2) + "," + \
                                      str(data[4]/2) + "," + \
                                      str(1)
                        else:
                            tmpdata = str(imageid) + "," + \
                                      str(data[1] - data[3] / 2.0) + "," + \
                                      str(data[2] - data[4] / 2.0) + "," + \
                                      str(data[3]) + "," + \
                                      str(data[4]) + "," + \
                                      str(1)
                        f_out.write(tmpdata + "\n")

                else:
                    data = [float(ll) for ll in line.split(' ')]
                    if data[5] >= 0.25:
                        tmpdata = str(imageid) + "," + \
                                  str((data[1] - data[3] / 2.0) * width) + "," + \
                                  str((data[2] - data[4] / 2.0) * height) + "," + \
                                  str(data[3] * width) + "," + \
                                  str(data[4] * height) + "," + \
                                  str(1)
                        f_out.write(tmpdata + "\n")

        imageid += 1  # label id+1