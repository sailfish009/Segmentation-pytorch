##DABNet
#python train.py --model DABNet --dataset cityscapes --train_val 'True' --max_epochs 200 --lr 0.001 --batch_size 8
# python train.py --model DABNet --dataset paris  --train_val 'True' --max_epochs 400 --lr 0.01 \
# --batch_size 16 --resume  /media/ding/Study/graduate/DABNet/checkpoint/paris/DABNetbs16gpu1_train/2020-03-30_13:10:41/model_161.pth

##FCN
python train.py --model FCN_8S_res18 --dataset paris --max_epochs 300 --train_val 'True' --val_epochs 40 --lr 0.0002 --batch_size 8

##UNet
#python train.py --model UNet --dataset paris --max_epochs 400 --train_val 'True' --lr 0.0002 --val_epochs 40 --batch_size 4
#python train.py --model UNet_res18_ori --dataset road --max_epochs 400 --train_val 'True' --val_epochs 40 --lr 0.0002 --batch_size 16
#python train.py --model UNet_res50 --dataset pairs --max_epochs 100 --train_val 'True' --val_epochs 50 --lr 0.0001 --batch_size 2
#python train.py --model UNet_overlap --dataset paris --max_epochs 200 --lr 0.001 --batch_size 4

##PSPNet_res50
#python train.py --model PSPNet_res50 --dataset cityscapes --max_epochs 300 --train_val 'True' --val_epochs 50 --lr 0.001 --batch_size 2 --num_loss 2

##BiSeNet
#python train.py --model BiSeNet_res18 --dataset paris --max_epochs 400 --train_val 'True' --val_epochs 40 --lr 0.0002 --batch_size 16

##ENet
#python train.py --model ENet --dataset paris --max_epochs 200 --lr 0.01 --batch_size 16

##GLAD
#python train.py --model GALD_res50 --dataset paris --max_epochs 200 --lr 0.01 --batch_size 4