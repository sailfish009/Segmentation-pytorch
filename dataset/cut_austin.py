# 导入相关的库
from PIL import Image
import glob
from tqdm import tqdm


#austin分割数据集
# '''执行之前确认以下两个dir的路径是否正确'''

input_dir = "/media/ding/Data/datasets/austin/Austin/AerialImageDataset/train/images/" # 需要裁剪的大图目录
output_dir = "/media/ding/Data/datasets/austin"  # 裁剪好的小图目录

# image_dir = "/media/ding/Data/datasets/paris/paris/*_image.png"
image_dir = input_dir +"/*.tif"
image_list = glob.glob(image_dir)
image_list.sort()
num_data = len(image_list)
image_list_train, image_list_val, image_list_test = image_list[:int(0.8*num_data)], image_list[int(0.8*num_data):int(0.85*num_data)], image_list[int(0.85*num_data):]
# print(image_list[0].split('/')[-1].split('.')[0] + '_gray.png')
# print(image_list[:3])
def cut_image(mode = 'train', image_list = None):
    for austin in tqdm(image_list):
        image_num = austin.split('/')[-1].split('.')[0].split('n')[-1]
        # print(austin.split('/')[-1].split('.')[0].split('n')[-1])
        # print("/media/ding/Files/ubuntu_study/Graduate/datasets/austin/" + austin.split('/')[-1].replace('image', 'labels'))
        label_path = input_dir.replace('images', 'gt') + "/" + austin.split('/')[-1].replace('.tif', '_gray.png')
        # print(label_path)
        # 打开一张图
        img = Image.open(austin)
        label = Image.open(label_path)
        # 图片尺寸
        size = img.size
        cropsize = 512
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
                crop_image = img.crop(box)
                crop_label = label.crop(box)
                # print(austin)
                # print(output_dir + '/{}/{}_image/austin{}_{}_{}.tif'.format(mode, cropsize, image_num, j, i))
                crop_image.save(output_dir + '/{}/{}_image/austin{}_{}_{}.png'.format(mode, cropsize, image_num, j, i))
                crop_label.save(output_dir + '/{}/{}_label/austin{}_{}_{}.png'.format(mode, cropsize, image_num, j, i))

if __name__ == '__main__':
    cut_image('train', image_list_train)
    cut_image('val', image_list_val)
    cut_image('test', image_list_test)