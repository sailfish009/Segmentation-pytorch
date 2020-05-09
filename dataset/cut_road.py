# 导入相关的库
from PIL import Image
import glob
from tqdm import tqdm


'''执行之前确认以下两个dir的路径是否正确'''
input_dir = "/media/ding/Data/datasets/massachusetts-roads-dataset/road_segmentation_ideal/training/output" # 需要裁剪的大图目录
output_dir = "/media/ding/Data/datasets/massachusetts-roads-dataset"  # 裁剪好的小图目录

# image_dir = "/media/ding/Data/datasets/paris/paris/*_image.png"
image_dir = input_dir + "/img-*_gray.png"
image_list = glob.glob(image_dir)
image_list.sort()
# num_data = len(image_list)

image_val_dir = input_dir.replace('training', 'testing') + "/img-*_gray.png"
image_val_list = glob.glob(image_val_dir)
image_val_list.sort()
print(image_val_list)
image_list_train, image_list_val, image_list_test = image_list[:], image_val_list[:], image_val_list[:]
# print(image_list[0].split('/')[-1].split('.')[0] + '_gray.png')
# print(image_list[:3])
def cut_image(mode = 'train', image_list = None):
    for road_label in tqdm(image_list):
        image_num = road_label.split('/')[-1].split('.')[0].split('-')[-1].split('_')[0]
        # print(road_label.split('/')[-1].split('_')[0].split('s')[-1])
        # print(input_dir.replace('output', 'input') + "/" + road_label.split('/')[-1].replace('_gray.png', '.png'))
        if not mode == 'train':
            image_path = input_dir.replace('output', 'input').replace('training', 'testing') + "/" + road_label.split('/')[-1].replace('_gray.png', '.png')
        else:
            image_path = input_dir.replace('output', 'input') + "/" + road_label.split('/')[-1].replace('_gray.png', '.png')

        # 打开一张图
        label = Image.open(road_label)
        image = Image.open(image_path)
        '''需要设置裁剪的尺寸'''
        # 图片尺寸
        size = label.size
        cropsize = 500
        stride = 400

        if (size[0] % stride > 0):
            num_w = int(size[0] / stride) + 1
        if ((stride * (num_w-2) + cropsize) > size[0]):
            num_w = int(size[0] / stride)   # 防止最后一个滑窗溢出，重复计算
        if (size[1] % stride > 0):
            num_h = int(size[1] / stride) + 1
        if stride * (num_h-2) + cropsize > size[1]:
            num_h = int(size[1] / stride)   # 防止最后一个滑窗溢出，重复计算
        for j in range(num_h):
            for i in range(num_w):
                x1 = int(i * stride)	#起始位置x1 = 0 * 513 = 0   0*400
                y1 = int(j * stride)	#		 y1 = 0 * 513 = 0   0*400
                x2 = min(x1 + cropsize, size[0])	# 末位置x2 = min(0+512, 3328)
                y2 = min(y1 + cropsize, size[1])   #	   y2 = min(0+512, 3072)
                x1 = max(int(x2 - cropsize), 0)  #重新校准起始位置x1 = max(512-512, 0)
                y1 = max(int(y2 - cropsize), 0)  #				  y1 = max(512-512, 0)
                box = (x1, y1, x2, y2)
                crop_label = label.crop(box)
                crop_image = image.crop(box)
                # print(road_label)
                # print(output_dir + '/{}/{}_image/road{}_{}_{}.png'.format(mode, cropsize, image_num, j, i))
                crop_image.save(output_dir + '/{}/{}_image/road{}_{}_{}.png'.format(mode, cropsize, image_num, j, i))
                crop_label.save(output_dir + '/{}/{}_label/road{}_{}_{}.png'.format(mode, cropsize, image_num, j, i))


if __name__ == '__main__':
    cut_image('train', image_list_train)
    cut_image('val', image_list_val)
    cut_image('test', image_list_test)


