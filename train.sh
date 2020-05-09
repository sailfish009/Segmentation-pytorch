##DABNet
#python train.py --model DABNet --dataset camvid --train_val 'True' --val_epochs 1 --max_epochs 400 --lr 0.0002 --batch_size 16
# python train.py --model DABNet --dataset paris  --train_val 'True' --max_epochs 400 --lr 0.0002 --batch_size 16

##FCN
#python train.py --model FCN_8S_res18 --dataset paris --max_epochs 200 --train_val 'True' --val_epochs 1 --lr 0.0005 --batch_size 4
#python train.py --model FCN_8S_res18 --dataset camvid --max_epochs 400 --train_val 'True' --val_epochs 1 --lr 0.0002 --batch_size 6
python train.py --model FCN_8S_res50 --dataset paris --max_epochs 100 --train_val 'True' --val_epochs 1 --lr 0.0008 --batch_size 1

##UNet
#python train.py --model UNet --dataset paris --max_epochs 400 --train_val 'True' --lr 0.0002 --val_epochs 1 --batch_size 4
#python train.py --model UNet_res18_ori --dataset paris --max_epochs 400 --train_val 'True' --val_epochs 1 --lr 0.0002 --batch_size 16
#python train.py --model UNet_res50_ori --dataset paris --max_epochs 400 --train_val 'True' --val_epochs 1 --lr 0.0002 --batch_size 8
#python train.py --model UNet_res18 --dataset paris --max_epochs 400 --train_val 'True' --val_epochs 1 --lr 0.0002 --batch_size 4
#python train.py --model UNet_res50 --dataset paris --max_epochs 400 --train_val 'True' --val_epochs 1 --lr 0.0002 --batch_size 2
#python train.py --model UNet_overlap --dataset paris --max_epochs 200 --lr 0.001 --batch_size 4

##PSPNet_res50
#python train.py --model PSPNet_res50 --dataset paris --max_epochs 400 --train_val 'True' --val_epochs 2 --lr 0.0001 --batch_size 4 --num_loss 2
#python train.py --model PSPNet_res50 --dataset camvid --max_epochs 400 --train_val 'True' --val_epochs 2 --lr 0.0002 --batch_size 4 --num_loss 2

##BiSeNet
#python train.py --model BiSeNet_res18 --dataset paris --max_epochs 400 --train_val 'True' --val_epochs 1 --lr 0.0002 --batch_size 16
#python train.py --model BiSeNet_res101 --dataset paris --max_epochs 400 --train_val 'True' --val_epochs 1 --lr 0.0002 --batch_size 8
#python train.py --model BiSeNet_res101 --dataset cityscapes --max_epochs 400 --train_val 'True' --val_epochs 1 --lr 0.001 --batch_size 4

##ENet
#python train.py --model ENet --dataset paris --max_epochs 200 --lr 0.01 --batch_size 16

##GLAD
#python train.py --model GALD_res50 --dataset paris --max_epochs 400 --train_val 'True' --val_epochs 1 --lr 0.0002 --batch_size 1