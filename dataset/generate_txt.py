import glob
def generate_txt(mode = 'train'):
    filename_list = glob.glob('/media/ding/Data/datasets/paris/512_image_107/overlap/{}/512_image/*.png'.format(mode))
    filename_list.sort()
    # print(filename_list)
    if mode == 'train':
        with open('./paris/paris_{}_list.txt'.format(mode), 'w+') as f:
            for filename in filename_list[:]:
                filename_gt = filename.replace('512_image', '512_label')
                # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
                # print(filename_gt)
                f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))

    elif mode == 'val':
        with open('./paris/paris_{}_list.txt'.format(mode), 'w+') as f:
            for filename in filename_list[:]:
                filename_gt = filename.replace('512_image', '512_label')
                # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
                # print(filename_gt)
                f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))

    else:
        with open('./paris/paris_{}_list.txt'.format(mode), 'w+') as f:
            for filename in filename_list:
                filename_gt = filename.replace('512_image', '512_label')
                # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
                # print(filename_gt)
                f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))

##austin分割数据集
# def generate_txt(mode = 'train'):
#     filename_list = glob.glob('/media/ding/Data/datasets/遥感数据集/遥感房屋分割数据集/austin/{}/512_image/*.png'.format(mode))
#     filename_list.sort()
#     # print(filename_list)
#     if mode == 'train':
#         with open('./austin/austin_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list[:8000]:
#                 filename_gt = filename.replace('512_image', '512_label')
#                 # print('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))
#
#     elif mode == 'val':
#         with open('./austin/austin_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list[:800]:
#                 filename_gt = filename.replace('512_image', '512_label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))
#
#     else:
#         with open('./austin/austin_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list:
#                 filename_gt = filename.replace('512_image', '512_label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))


# def generate_txt(mode = 'train'):
#     filename_list = glob.glob('/media/ding/Data/datasets/road/{}/500_image/*.png'.format(mode))
#     filename_list.sort()
#     # print(filename_list)
#     if mode == 'train':
#         with open('./road/road_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list[:]:
#                 filename_gt = filename.replace('500_image', '500_label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))
#
#     elif mode == 'val':
#         with open('./road/road_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list[:]:
#                 filename_gt = filename.replace('500_image', '500_label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))
#
#     else:
#         with open('./road/road_{}_list.txt'.format(mode), 'w+') as f:
#             for filename in filename_list:
#                 filename_gt = filename.replace('500_image', '500_label')
#                 # print(filename.split('/')[-2]+'/'+train.split('/')[-1])
#                 # print(filename_gt)
#                 f.write('{}/{}\t{}/{}\n'.format(mode, filename.split('/')[-2]+'/'+filename.split('/')[-1], mode, filename_gt.split('/')[-2]+'/'+filename_gt.split('/')[-1]))
if __name__ == '__main__':
    generate_txt('train')
    generate_txt('val')
    generate_txt('test')
    print('Finsh!')
