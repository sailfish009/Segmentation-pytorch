# 导入相关的库
from PIL import Image
import glob
from tqdm import tqdm

'''执行之前确认以下两个dir的路径是否正确'''
input_dir = "/media/ding/Data/datasets/paris/paris_origin_107" # 需要裁剪的大图目录
output_dir = "/media/ding/Data/datasets/paris/512_image_107"  # 裁剪好的小图目录

# image_dir = "/media/ding/Data/datasets/paris/paris/*_image.png"
image_dir = input_dir +"/*_image.png"
image_list = glob.glob(image_dir)
image_list.sort()
num_data = len(image_list)
image_list_train, image_list_val, image_list_test = image_list[:int(0.8*num_data)], image_list[int(0.8*num_data):], image_list[int(0.8*num_data):]
# print(image_list[0].split('/')[-1].split('.')[0] + '_gray.png')
# print(image_list[:3])
def cut_image(mode = 'train', image_list = None):
    for paris in tqdm(image_list):
        image_num = paris.split('/')[-1].split('_')[0].split('s')[-1]
        # print(paris.split('/')[-1].split('_')[0].split('s')[-1])
        # print("/media/ding/Files/ubuntu_study/Graduate/datasets/paris/" + paris.split('/')[-1].replace('image', 'labels'))
        label_path = input_dir + "/" + paris.split('/')[-1].replace('image', 'labels_gray')
        # 打开一张图
        img = Image.open(paris)
        label = Image.open(label_path)
        # 图片尺寸
        size = img.size
        cropsize = 512

        if(size[0] % cropsize > 0):
            num_w = int(size[0] / cropsize) + 1
        else:
            num_w = size[0] // cropsize
        if(size[1] % cropsize > 0):
            num_h = int(size[1] / cropsize) + 1
        else:
            num_h = size[1] // cropsize
        # print(num_w, num_h)

        for j in range(num_h):
            for i in range(num_w):
                if(size[0] % cropsize> 0 and i == num_w-1):
                    box = (size[0] - cropsize, cropsize * j, size[0], cropsize * (j + 1))
                elif(size[1] % cropsize > 0 and j == num_h - 1):
                    box = (cropsize * i, size[1]-cropsize, cropsize * (i + 1), size[1])
                else:
                    box = (cropsize * i, cropsize * j, cropsize * (i + 1), cropsize * (j + 1))
                crop_image = img.crop(box)
                crop_label = label.crop(box)
                # print(paris)
                # print('paris/paris{}_{}_{}.png'.format(image_num, j, i))
                crop_image.save(output_dir + '/{}/{}_image/paris{}_{}_{}.png'.format(mode, cropsize, image_num, j, i))
                crop_label.save(output_dir + '/{}/{}_label/paris{}_{}_{}.png'.format(mode, cropsize, image_num, j, i))


#austin分割数据集
# '''执行之前确认以下两个dir的路径是否正确'''

# input_dir = "/media/ding/Data/datasets/遥感数据集/遥感房屋分割数据集/AerialImageDataset/train/images" # 需要裁剪的大图目录
# output_dir = "/media/ding/Data/datasets/austin"  # 裁剪好的小图目录
#
# # image_dir = "/media/ding/Data/datasets/paris/paris/*_image.png"
# image_dir = input_dir +"/*.tif"
# image_list = glob.glob(image_dir)
# image_list.sort()
# num_data = len(image_list)
# image_list_train, image_list_val, image_list_test = image_list[:int(0.8*num_data)], image_list[int(0.8*num_data):int(0.85*num_data)], image_list[int(0.85*num_data):]
# # print(image_list[0].split('/')[-1].split('.')[0] + '_gray.png')
# # print(image_list[:3])
# def cut_image(mode = 'train', image_list = None):
#     for austin in tqdm(image_list):
#         image_num = austin.split('/')[-1].split('.')[0].split('n')[-1]
#         # print(austin.split('/')[-1].split('.')[0].split('n')[-1])
#         # print("/media/ding/Files/ubuntu_study/Graduate/datasets/austin/" + austin.split('/')[-1].replace('image', 'labels'))
#         label_path = input_dir.replace('images', 'gt') + "/" + austin.split('/')[-1].replace('.tif', '_gray.png')
#         # print(label_path)
#         # 打开一张图
#         img = Image.open(austin)
#         label = Image.open(label_path)
#         # 图片尺寸
#         size = img.size
#         cropsize = 512
#
#         if(size[0] % cropsize > 0):
#             num_w = int(size[0] / cropsize) + 1
#         else:
#             num_w = size[0] // cropsize
#         if(size[1] % cropsize > 0):
#             num_h = int(size[1] / cropsize) + 1
#         else:
#             num_h = size[1] // cropsize
#         # print(num_w, num_h)
#
#         for j in range(num_h):
#             for i in range(num_w):
#                 if(size[0] % cropsize> 0 and i == num_w-1):
#                     box = (size[0] - cropsize, cropsize * j, size[0], cropsize * (j + 1))
#                 elif(size[1] % cropsize > 0 and j == num_h - 1):
#                     box = (cropsize * i, size[1]-cropsize, cropsize * (i + 1), size[1])
#                 else:
#                     box = (cropsize * i, cropsize * j, cropsize * (i + 1), cropsize * (j + 1))
#                 crop_image = img.crop(box)
#                 crop_label = label.crop(box)
#                 # print(austin)
#                 # print(output_dir + '/{}/{}_image/austin{}_{}_{}.tif'.format(mode, cropsize, image_num, j, i))
#                 crop_image.save(output_dir + '/{}/{}_image/austin{}_{}_{}.png'.format(mode, cropsize, image_num, j, i))
#                 crop_label.save(output_dir + '/{}/{}_label/austin{}_{}_{}.png'.format(mode, cropsize, image_num, j, i))


'''执行之前确认以下两个dir的路径是否正确'''
# input_dir = "/media/ding/Data/datasets/massachusetts-roads-dataset/road_segmentation_ideal/training/output" # 需要裁剪的大图目录
# output_dir = "/media/ding/Data/datasets/massachusetts-roads-dataset"  # 裁剪好的小图目录
#
# # image_dir = "/media/ding/Data/datasets/paris/paris/*_image.png"
# image_dir = input_dir + "/img-*_gray.png"
# image_list = glob.glob(image_dir)
# image_list.sort()
# # num_data = len(image_list)
#
# image_val_dir = input_dir.replace('training', 'testing') + "/img-*_gray.png"
# image_val_list = glob.glob(image_val_dir)
# image_val_list.sort()
# print(image_val_list)
# image_list_train, image_list_val, image_list_test = image_list[:], image_val_list[:], image_val_list[:]
# # print(image_list[0].split('/')[-1].split('.')[0] + '_gray.png')
# # print(image_list[:3])
# def cut_image(mode = 'train', image_list = None):
#     for road_label in tqdm(image_list):
#         image_num = road_label.split('/')[-1].split('.')[0].split('-')[-1].split('_')[0]
#         # print(road_label.split('/')[-1].split('_')[0].split('s')[-1])
#         # print(input_dir.replace('output', 'input') + "/" + road_label.split('/')[-1].replace('_gray.png', '.png'))
#         if not mode == 'train':
#             image_path = input_dir.replace('output', 'input').replace('training', 'testing') + "/" + road_label.split('/')[-1].replace('_gray.png', '.png')
#         else:
#             image_path = input_dir.replace('output', 'input') + "/" + road_label.split('/')[-1].replace('_gray.png', '.png')
#
#         # 打开一张图
#         label = Image.open(road_label)
#         image = Image.open(image_path)
#         '''需要设置裁剪的尺寸'''
#         # 图片尺寸
#         size = label.size
#         cropsize = 500
#
#         if(size[0] % cropsize > 0):
#             num_w = int(size[0] / cropsize) + 1
#         else:
#             num_w = size[0] // cropsize
#         if(size[1] % cropsize > 0):
#             num_h = int(size[1] / cropsize) + 1
#         else:
#             num_h = size[1] // cropsize
#         # print(num_w, num_h)
#
#         for j in range(num_h):
#             for i in range(num_w):
#                 if(size[0] % cropsize> 0 and i == num_w-1):
#                     box = (size[0] - cropsize, cropsize * j, size[0], cropsize * (j + 1))
#                 elif(size[1] % cropsize > 0 and j == num_h - 1):
#                     box = (cropsize * i, size[1]-cropsize, cropsize * (i + 1), size[1])
#                 else:
#                     box = (cropsize * i, cropsize * j, cropsize * (i + 1), cropsize * (j + 1))
#                 crop_label = label.crop(box)
#                 crop_image = image.crop(box)
#                 # print(road_label)
#                 # print(output_dir + '/{}/{}_image/road{}_{}_{}.png'.format(mode, cropsize, image_num, j, i))
#                 crop_image.save(output_dir + '/{}/{}_image/road{}_{}_{}.png'.format(mode, cropsize, image_num, j, i))
#                 crop_label.save(output_dir + '/{}/{}_label/road{}_{}_{}.png'.format(mode, cropsize, image_num, j, i))


if __name__ == '__main__':
    cut_image('train', image_list_train)
    cut_image('val', image_list_val)
    cut_image('test', image_list_test)

###单张测试代码
# img = Image.open('/media/ding/Data/datasets/austin/Austin/AerialImageDataset/train/images/austin1.tif')
# label = Image.open('/media/ding/Data/datasets/austin/Austin/AerialImageDataset/train/gt/austin1.tif')
# # 图片尺寸
# size = img.size
# cropsize = 512
#
# if(size[0] % cropsize > 0):
#     num_w = int(size[0] / cropsize) + 1
# else:
#     num_w = size[0] // cropsize
# if(size[1] % cropsize > 0):
#     num_h = int(size[1] / cropsize) + 1
# else:
#     num_h = size[1] // cropsize
# # print(num_w, num_h)
#
# for j in range(num_h):
#     for i in range(num_w):
#         if(size[0] % cropsize> 0 and i == num_w-1):
#             box = (size[0] - cropsize, cropsize * j, size[0], cropsize * (j + 1))
#         elif(size[1] % cropsize > 0 and j == num_h - 1):
#             box = (cropsize * i, size[1]-cropsize, cropsize * (i + 1), size[1])
#         else:
#             box = (cropsize * i, cropsize * j, cropsize * (i + 1), cropsize * (j + 1))
#         crop_image = img.crop(box)
#         crop_label = label.crop(box)
#         # print(austin)
#         # print(output_dir + '/{}/{}_image/austin{}_{}_{}.tif'.format(mode, cropsize, image_num, j, i))
#         crop_image.save('austin{}_{}.tif'.format( j, i))
#         crop_label.save('austinl{}_{}.tif'.format( j, i))

