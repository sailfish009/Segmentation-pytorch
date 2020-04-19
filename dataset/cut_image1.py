import random
random.seed(1)
import tifffile as tif

import cv2
import numpy as np
import random

from tqdm import tqdm
import pandas as pd
import os

size = 256
represent_train_path = '/media/ding/Files/ubuntu_study/Graduate/datasets/'


# 随机窗口采样
def generate_train_dataset(image_num=1000,
                           train_image_path=represent_train_path + 'cut_image/',
                           train_label_path=represent_train_path + 'cut_label/'):
    '''
    该函数用来生成训练集，切图方法为随机切图采样
    :param image_num: 生成样本的个数
    :param train_image_path: 切图保存样本的地址
    :param train_label_path: 切图保存标签的地址
    :return:
    '''

    # 用来记录所有的子图的数目
    g_count = 1
    images_path = []
    for i in range(1, 81):
        if i == 15 or i == 16 or i == 17 or i == 18 or i == 19 or i == 20 or i == 22 or i == 77 or i == 78 or i == 75 or i == 80:
            continue
        elif i == 1 or i == 8 or i == 21 or i == 22 or i == 23 or i == 27 or i == 30 or i == 35 or i == 36 or i == 44 or i == 53 or i == 64 or i == 68:
            continue
        else:
            images_path.append('D:\\dataset_chain\water\DeeCamp-26-耕地\yuanyang\{0}\l{0}.tif'.format(i))

    labels_path = []
    for i in range(1, 81):
        if i == 15 or i == 16 or i == 17 or i == 18 or i == 19 or i == 20 or i == 22 or i == 77 or i == 78 or i == 75 or i == 80:
            continue
        elif i == 1 or i == 8 or i == 21 or i == 22 or i == 23 or i == 27 or i == 30 or i == 35 or i == 36 or i == 44 or i == 53 or i == 64 or i == 68:
            continue
        else:
            labels_path.append('D:\dataset_chain\water\DeeCamp-26-耕地\yuanyang\{0}\l{0}_mask.tif'.format(i))
        #    这是test的读取方式
    #     labels_path = []
    #     test_index = [1,8,21,22,23,27,30,35,36,44,53,64,68,75,80]
    #     for i in range(15):
    #         b = test_index[i]
    #         labels_path.append('D:\dataset_chain\water\DeeCamp-26-耕地\yuanyang\{0}\l{0}_mask.tif'.format(b))
    #         labels_path.append("({0},:)".format(b))

    # 每张图片生成子图的个数
    image_each = image_num // len(images_path)
    image_path, label_path = [], []
    for i in tqdm(range(len(images_path))):
        count = 0
        tif = TIFF.open(images_path[i], mode='r')
        image = tif.imread(images_path[i]).astype(np.uint8)

        label = tif.imread(labels_path[i]).astype(np.uint8)
        X_height, X_width = image.shape[0], image.shape[1]
        while count < image_each:
            random_width = random.randint(0, X_width - size - 1)
            random_height = random.randint(0, X_height - size - 1)
            image_ogi = image[random_height: random_height + size, random_width: random_width + size, :]
            label_ogi = label[random_height: random_height + size, random_width: random_width + size]

            #             image_d, label_d = data_augment(image_ogi, label_ogi) #这里是图像增强

            image_d, label_d = image_d, label_d
            image_path.append(train_image_path + '%05d.png' % g_count)
            label_path.append(train_label_path + '%05d.png' % g_count)
            tif = TIFF.open('filename.tif', mode='w')
            tif.imsave((train_image_path + '%05d.png' % g_count), image_d)
            tif.imsave((train_label_path + '%05d.png' % g_count), label_d)

            count += 1
            g_count += 1
    df = pd.DataFrame({'image': image_path, 'label': label_path})
    df.to_csv('dataset/path_list.csv', index=False)