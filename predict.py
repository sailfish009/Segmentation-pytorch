import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from argparse import ArgumentParser
from tqdm import tqdm
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from tools.utils import save_predict
from tools.convert_state import convert_state_dict
from tools.SegmentationMetric import SegmentationMetric


def predict(args, test_loader, model):
    """
    args:
      test_loader: loaded for test dataset, for those that do not provide label on the test set
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    total_batches = len(test_loader)
    pbar = tqdm(iterable=enumerate(test_loader), total=total_batches, desc='Predicting')
    Miou_list = []
    Iou_list = []
    Pa_list = []
    Mpa_list = []
    Fmiou_list = []
    for i, (input, gt, size, name) in pbar:
        with torch.no_grad():
            input_var = Variable(input).cuda()
        output = model(input_var)
        torch.cuda.synchronize()
        output = output.cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        gt = np.asarray(gt[0], dtype=np.uint8)

        # 计算miou
        metric = SegmentationMetric(numClass=args.classes)
        metric.addBatch(imgPredict=output, imgLabel=gt)
        miou, iou = metric.meanIntersectionOverUnion()
        fmiou = metric.Frequency_Weighted_Intersection_over_Union()
        pa = metric.pixelAccuracy()
        mpa = metric.meanPixelAccuracy()
        Miou_list.append(miou)
        Fmiou_list.append(fmiou)
        Pa_list.append(pa)
        Mpa_list.append(mpa)
        iou = np.array(iou)
        Iou_list.append(iou)

        # Save the predict greyscale output for Cityscapes official evaluation
        # Modify image name to meet official requirement
        '''设置输出原图和预测图片的颜色灰度还是彩色'''
        save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
                     output_grey=False, output_color=True, gt_color=False)
    miou = np.mean(Miou_list)
    fmiou = np.mean(Fmiou_list)
    pa = np.mean(Pa_list)
    mpa = np.mean(Mpa_list)
    Iou_list = np.asarray(Iou_list)
    iou = np.mean(Iou_list, axis=0)
    cls_iu = dict(zip(range(args.classes), iou))
    return miou, cls_iu, fmiou, pa, mpa



def test_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(args.model, num_classes=args.classes)

    if args.cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    if not os.path.exists(args.save_seg_dir):
        os.makedirs(args.save_seg_dir)

    # load the test set
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers, none_gt=True)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=====> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    print("=====> beginning testing")
    print("test set length: ", len(testLoader))
    miou, class_iou, fmiou, pa, mpa = predict(args, testLoader, model)
    print('Miou is: {:.4f}\nClass iou is: {}\nFMiou is: {:.4f}\nPa is: {:.4f}\nMpa is: {:.4f}'.format(miou, class_iou,
                                                                                                      fmiou, pa, mpa))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="DABNet", help="model name: Context Guided Network (CGNet)")
    parser.add_argument('--dataset', default="paris", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=1, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,
                        default=None,
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./result/",
                        help="saving path of prediction result")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    # args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.model)
    save_dirname = args.checkpoint.split('/')[-3] + '_' + args.checkpoint.split('/')[-1].split('.')[0]
    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, 'predict', save_dirname)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    elif args.dataset == 'paris':
        args.classes = 3
    elif args.dataset == 'austin':
        args.classes = 2
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    test_model(args)
