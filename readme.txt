准备数据集:paris
数据结构:
|--paris-origin
|       |--label*.png
|       |--image*.png
|--paris
|    |--train
|       |--512-label
|           |--label*.png
|           |--image*.png
|       |--512-image 
|           |--label*.png
|           |--image*.png    
|    |--val
|       |--512-label
|           |--label*.png
|           |--image*.png
|       |--512-image 
|           |--label*.png
|           |--image*.png    
|    |--test
|       |--512-label
|           |--label*.png
|           |--image*.png
|       |--512-image
|           |--label*.png
|           |--image*.png      
paris
1.datasets文件夹下change_colour.py,将彩色的label图片变成灰度图,且颜色从0开始计数
2.datasets文件夹下cut_image2.py,将原始3000尺寸的大圩切分为自己需要的512尺寸图片
3.datasets文件夹下gengrate_txt.py,生成切割小图的路径写入txt文件

训练
train.sh脚本,修改参数即可
builders文件夹下dataset_builder.py文件的data_dir需要修改为数据集的文件夹目录

预测两种方法
1.predict.sh脚本--预测小图,修改--checkpoint等参数;接着result/concat_image.py拼接成大图
2.predict_sliding.sh脚本--滑动窗口预测大图
