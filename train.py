import os
import time
import torch
import torch.nn as nn
import timeit
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_train
from tools.utils import setup_seed, init_weight, netParams
from tools.metric import get_iou
from tools.loss import CrossEntropyLoss2d, ProbOhemCrossEntropy2d
from tools.lr_scheduler import WarmupPolyLR, poly_learning_rate
from tqdm import tqdm
from tools.convert_state import convert_state_dict
from time import strftime, localtime
date = strftime("%Y-%m-%d_%H:%M:%S", localtime())
print(date)

GLOBAL_SEED = 1234


def val(args, val_loader, model):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    # evaluation mode
    model.eval()
    total_batches = len(val_loader)

    data_list = []
    pbar = tqdm(iterable=enumerate(val_loader), total=total_batches, desc='Val')
    for i, (input, label, size, name) in pbar:
        with torch.no_grad():
            input_var = Variable(input).cuda()
        output = model(input_var)

        output = output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append([gt.flatten(), output.flatten()])

    meanIoU, per_class_iu = get_iou(data_list, args.classes, save_path=None)
    return meanIoU, per_class_iu

def train(args, train_loader, model, criterion, optimizer, epoch):
    """
    args:
       train_loader: loaded for training dataset
       model: model
       criterion: loss function
       optimizer: optimization algorithm, such as ADAM or SGD
       epoch: epoch number
    return: average loss, per class IoU, and mean IoU
    """
    model.train()
    epoch_loss = []

    total_batches = len(train_loader)

    st = time.time()
    pbar = tqdm(iterable=enumerate(train_loader), total=total_batches, desc='Epoch {}'.format(epoch))
    for iteration, batch in pbar:
        args.per_iter = total_batches
        args.max_iter = args.max_epochs * args.per_iter
        args.cur_iter = epoch * args.per_iter + iteration
        scheduler = WarmupPolyLR(optimizer, T_max=args.max_iter, cur_iter=args.cur_iter, warmup_factor=1.0 / 3,
                                 warmup_iters=500, power=0.9)
        lr = optimizer.param_groups[0]['lr']

        images, labels, _, _ = batch
        images = Variable(images).cuda()
        labels = Variable(labels.long()).cuda()

        # single loss
        if args.num_loss == 1:
            # print(images.size())
            output = model(images)
            # print(labels.size())
            # print('output shape is',output.shape)
            loss = criterion(output, labels)

        # muti loss
        else:
            output = model(images, labels)
            loss1 = criterion(output[0], labels)
            loss2 = output[1]
            loss = loss1 + loss2

        pbar.set_postfix(lr='%.5f' % lr, loss='%.5f' % loss)

        scheduler.step()
        optimizer.zero_grad()  # set the grad to zero
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

    # pbar.close()
    time_taken_epoch = time.time() - st
    remain_time = time_taken_epoch * (args.max_epochs - 1 - epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    print("Remaining training time = %d hour %d minutes %d seconds" % (h, m, s))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr

def train_model(args):
    """
    args:
       args: global arguments
    """

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("=====> input size:{}".format(input_size))

    print(args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    # set the seed
    setup_seed(GLOBAL_SEED)
    print("=====> set Global Seed: ", GLOBAL_SEED)

    cudnn.enabled = True

    # build the model and initialization
    model = build_model(args.model, num_classes=args.classes)
    init_weight(model, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-3, 0.1,
                mode='fan_in')

    print("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

    # load data and data augmentation
    datas, trainLoader, valLoader = build_dataset_train(args.dataset, input_size, args.batch_size, args.train_type,
                                                        args.random_scale, args.random_mirror, args.num_workers)

    print('=====> Dataset statistics')
    print("data['classWeights']: ", datas['classWeights'])
    print('mean and std: ', datas['mean'], datas['std'])

    # define loss function, respectively
    weight = torch.from_numpy(datas['classWeights'])

    if args.dataset == 'camvid':
        criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)
    elif args.dataset == 'cityscapes':
        min_kept = int(args.batch_size // len(args.gpus) * h * w // 16)
        criteria = ProbOhemCrossEntropy2d(use_weight=True, ignore_label=ignore_label,
                                          thresh=0.7, min_kept=min_kept)
    elif args.dataset == 'paris':
        criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)

        # criteria = nn.CrossEntropyLoss(weight=weight)

        # min_kept = int(args.batch_size // len(args.gpus) * h * w // 16)
        # criteria = ProbOhemCrossEntropy2d(ignore_label=ignore_label, thresh=0.7, min_kept=min_kept, use_weight=False)
    elif args.dataset == 'austin':
        criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)
        # criteria = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    if args.cuda:
        criteria = criteria.cuda()
        if torch.cuda.device_count() > 1:
            print("torch.cuda.device_count()=", torch.cuda.device_count())
            args.gpu_nums = torch.cuda.device_count()
            model = nn.DataParallel(model).cuda()  # multi-card data parallel
        else:
            args.gpu_nums = 1
            print("single GPU for training")
            model = model.cuda()  # 1-card data parallel

    args.savedir = (args.savedir + args.dataset + '/' + args.model + 'bs'
                    + str(args.batch_size) + 'gpu' + str(args.gpu_nums) + "_" + str(args.train_type) + '/' + str(date) +'/')

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    start_epoch = 0

    # continue training
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
            print("=====> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.resume))

    model.train()
    cudnn.benchmark = True

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s Seed: %s\n %s" % (str(total_paramters/ 1e6), GLOBAL_SEED, args))
        logger.write("\n%s\t%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s" % ('Epoch', 'lr', 'Loss(Tr)', 'mIOU (val)', 'Class-0', 'Class-1', 'Class-2'))
    logger.flush()

    # define optimization criteria
    if args.dataset == 'camvid':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=2e-4)

    elif args.dataset == 'cityscapes':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), args.lr, momentum=0.9, weight_decay=1e-4)

    elif args.dataset == 'paris':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), args.lr, momentum=0.9, weight_decay=1e-4)

    elif args.dataset == 'austin':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), args.lr, momentum=0.9, weight_decay=1e-4)
    lossTr_list = []
    epoches = []
    mIOU_val_list = []

    print('***********************************************\n'
          '*******        Begining traing          *******\n'
          '***********************************************')
    for epoch in range(start_epoch, args.max_epochs):
        # training
        lossTr, lr = train(args, trainLoader, model, criteria, optimizer, epoch)
        lossTr_list.append(lossTr)

        # validation
        if (epoch % args.val_epochs == 0 and args.train_val == 'True') or epoch == (args.max_epochs - 1) :
            epoches.append(epoch)
            mIOU_val, per_class_iu = val(args, valLoader, model)
            mIOU_val_list.append(mIOU_val)
            # record train information
            if args.classes == 3:
                logger.write("\n%d\t%.5f\t\t%.5f\t\t\t%.4f\t\t\t%0.4f\t\t%0.4f\t\t%0.4f" % (epoch, lr, lossTr, mIOU_val, per_class_iu[0], per_class_iu[1], per_class_iu[2]))
            elif args.classes == 2:
                logger.write("\n%d\t%.5f\t\t%.5f\t\t\t%.4f\t\t\t%0.4f\t\t%0.4f" % (epoch, lr, lossTr, mIOU_val, per_class_iu[0], per_class_iu[1]))
            else:
                for i in per_class_iu:
                    logger.write("\n%d\t\t%.5f\t\t%.5f\t\t\t%.4f\t\t\t%0.4f\t\t%0.4f" % (epoch, lr, lossTr, mIOU_val, per_class_iu[0], per_class_iu[1]))
            logger.flush()
            print("Epoch %d\tTrain Loss = %.4f\t mIOU(val) = %.4f\t lr= %.5f\n" % (epoch,
                                                                                 lossTr,
                                                                                 mIOU_val, lr))
        else:
            # record train information
            logger.write("\n%d\t%.5f\t\t%.5f" % (epoch, lr, lossTr))
            logger.flush()
            print("Epoch %d\tTrain Loss = %.4f\t lr= %.6f\n" % (epoch, lossTr, lr))

        # save the model
        model_file_name = args.savedir + '/model_' + str(epoch + 1) + '.pth'
        state = {"epoch": epoch, "model": model.state_dict()}

        if epoch >= args.max_epochs - args.val_epochs:
            torch.save(state, model_file_name)
        elif not epoch % args.save_epochs:
            torch.save(state, model_file_name)

        # draw plots for visualization
        if epoch % args.val_epochs == 0 or epoch == (args.max_epochs - 1):
            # Plot the figures per args.val_epochs epochs
            fig1, ax1 = plt.subplots(figsize=(11, 8))

            ax1.plot(range(start_epoch, epoch + 1), lossTr_list)
            ax1.set_title("Average training loss vs epochs")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Current loss")

            plt.savefig(args.savedir + "loss_vs_epochs.png")

            plt.clf()

            fig2, ax2 = plt.subplots(figsize=(11, 8))

            ax2.plot(epoches, mIOU_val_list, label="Val IoU")
            ax2.set_title("Average IoU vs epochs")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Current IoU")
            plt.legend(loc='lower right')

            plt.savefig(args.savedir + "iou_vs_epochs.png")

            plt.close('all')

    logger.close()


if __name__ == '__main__':
    start = timeit.default_timer()
    parser = ArgumentParser()
    parser.add_argument('--model', default="DABNet", help="model name in model_builder.py")
    parser.add_argument('--dataset', default="paris", help="dataset: paris austin cityscapes or camvid ")
    parser.add_argument('--train_type', type=str, default="train",
                        help="ontrain for training on train set, ontrainval for training on train+val set")
    parser.add_argument('--train_val', type=str, default='True', help="train with val")
    parser.add_argument('--max_epochs', type=int, default=300,
                        help="the number of epochs: 300 for train set, 350 for train+val set")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--batch_size', type=int, default=8, help="the batch size is set to 16 for 2 GPUs")
    parser.add_argument('--num_loss', type=int, default=1, help="the number of loss for train")
    parser.add_argument('--val_epochs', type=int, default=10, help=" the number of val of epochs")
    parser.add_argument('--save_epochs', type=int, default=1, help=" the number of save checkpoint of epochs")
    parser.add_argument('--random_mirror', type=bool, default=False, help="input image random mirror")
    parser.add_argument('--random_scale', type=bool, default=False, help="input image resize 0.5 to 2")
    parser.add_argument('--num_workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--savedir', default="./checkpoint/", help="directory to save the model snapshot")
    parser.add_argument('--resume', type=str, default="",
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="0", help="default GPU devices (0,1)")
    args = parser.parse_args()

    if args.dataset == 'cityscapes':
        args.classes = 19
        args.input_size = '512,1024'
        ignore_label = 255
    elif args.dataset == 'camvid':
        args.classes = 11
        args.input_size = '360,480'
        ignore_label = 11
    elif args.dataset == 'paris':
        args.classes = 3
        # args.input_size = '1024,1024'
        args.input_size = '512,512'
        ignore_label = 3
    elif args.dataset == 'austin':
        args.classes = 2
        # args.input_size = '1024,1024'
        args.input_size = '512,512'
        ignore_label = 2
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    train_model(args)
    end = timeit.default_timer()
    hour = 1.0 * (end - start) / 3600
    minute = (hour - int(hour)) * 60
    print("training time: %d hour %d minutes" % (int(hour), int(minute)))
