from PIL import Image
from tqdm import tqdm
import glob

i = 1
j = 1

#获取指定目录下的所有图片
# image_dir = "/media/ding/Data/datasets/pairs/paris/*labels.png"
# image_list = glob.glob(image_dir)
# print(len(image_list))
# print(image_list[0].split('/')[-1].split('.')[0] + '_gray.png')

image_dir = "/media/ding/Data/datasets/遥感数据集/遥感房屋分割数据集/AerialImageDataset/train/gt/*.tif"
image_list = glob.glob(image_dir)

for cnt,pic in tqdm(enumerate(image_list)):
    # print("/media/ding/Files/ubuntu_study/Graduate/datasets/label/" + image_list[cnt].split('/')[-1].split('.')[0] + '_gray.png')
    img = Image.open(pic).convert("L")
    width = img.size[0]#长度
    height = img.size[1]#宽度
    for i in range(0,width):#遍历所有长度的点
        for j in range(0,height):#遍历所有宽度的点
            data = (img.getpixel((i, j)))#打印该图片的所有点
            # print(data)#打印每个像素点的颜色RGBA的值(r,g,b,alpha)

            #paris分割数据集
            # if(data == 255):
            #     img.putpixel((i, j), 0)
            # if (data == 29):
            #     img.putpixel((i, j), 1)
            # if (data == 76):
            #     img.putpixel((i, j), 2)

            #austin房屋分割数据集
            # if(data == 0):
            #     img.putpixel((i, j), 0)
            # if(data == 255):
            #     img.putpixel((i, j), 1)

            #cityscapes分割数据集
            if(data == 255):
                img.putpixel((i, j), 0)


    # img.save("/media/ding/Data/datasets/pairs/paris/" + image_list[cnt].split('/')[-1].split('.')[0] + '_gray.png')#保存修改像素点后的图片
    img.save("/media/ding/Data/datasets/遥感数据集/遥感房屋分割数据集/AerialImageDataset/train/gt/" + image_list[cnt].split('/')[-1].split('.')[0] + '_gray.png')#保存修改像素点后的图片
