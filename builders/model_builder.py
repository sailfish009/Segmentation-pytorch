from model.DABNet import DABNet
from model.UNet import UNet
from model.UNet_overlap import UNet_overlap
from model.GALDNet import GALD_res50, GALD_res101
from model.ENet import ENet
from model.PSPNet import PSPNet
from model.UNet_res_ori import UNet_res_ori  #没改变resnet基本结构
from model.UNet_res import UNet_res
from model.FCN import FCN_res
from model.BiSeNet import BiSeNet



def build_model(model_name, num_classes):
    if model_name == 'DABNet':
        return DABNet(classes=num_classes)

    elif model_name == 'FCN_8S_res18':
        return FCN_res(backbone='resnet18', classes=num_classes, pretrained=True, scale=8)
    elif model_name == 'FCN_8S_res50':
        return FCN_res(backbone='resnet50', classes=num_classes, pretrained=True, scale=8)
    elif model_name == 'FCN_8S_res101':
        return FCN_res(backbone='resnet101', classes=num_classes, pretrained=True, scale=8)
    elif model_name == 'FCN_32S_res18':
        return FCN_res(backbone='resnet18', classes=num_classes, pretrained=True, scale=32)
    elif model_name == 'FCN_32S_res50':
        return FCN_res(backbone='resnet50', classes=num_classes, pretrained=True, scale=32)
    elif model_name == 'FCN_32S_res101':
        return FCN_res(backbone='resnet101', classes=num_classes, pretrained=True, scale=32)

    elif model_name == 'UNet_res18':
        return UNet_res(backbone='resnet18', pretrained=True, classes=num_classes)
    elif model_name == 'UNet_res34':
        return UNet_res(backbone='resnet34', pretrained=True, classes=num_classes)
    elif model_name == 'UNet_res50':
        return UNet_res(backbone='resnet50', pretrained=True, classes=num_classes)
    elif model_name == 'UNet_res101':
        return UNet_res(backbone='resnet101', pretrained=True, classes=num_classes)

    elif model_name == 'UNet_res18_ori':
        return UNet_res_ori(backbone='resnet18', pretrained=True, classes=num_classes)
    elif model_name == 'UNet_res34_ori':
        return UNet_res_ori(backbone='resnet34', pretrained=True, classes=num_classes)
    elif model_name == 'UNet_res50_ori':
        return UNet_res_ori(backbone='resnet50', pretrained=True, classes=num_classes)
    elif model_name == 'UNet_res101_ori':
        return UNet_res_ori(backbone='resnet101', pretrained=True, classes=num_classes)

    elif model_name == 'PSPNet_res18':
        return PSPNet(layers=18, bins=(1, 2, 3, 6), dropout=0.1, classes=num_classes, zoom_factor=8, use_ppm=True,
                            pretrained=True)
    elif model_name == 'PSPNet_res34':
        return PSPNet(layers=34, bins=(1, 2, 3, 6), dropout=0.1, classes=num_classes, zoom_factor=8, use_ppm=True,
                      pretrained=True)
    elif model_name == 'PSPNet_res50':
        return PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=num_classes, zoom_factor=8, use_ppm=True,
                      pretrained=True)
    elif model_name == 'PSPNet_res101':
        return PSPNet(layers=101, bins=(1, 2, 3, 6), dropout=0.1, classes=num_classes, zoom_factor=8, use_ppm=True,
                      pretrained=True)

    ## backbone == vgg
    elif model_name == 'UNet':
        return UNet(classes=num_classes)
    elif model_name == 'UNet_overlap':
        return UNet_overlap(classes=num_classes)

    elif model_name == 'BiSeNet_res18':
        return  BiSeNet(backbone='resnet18', n_classes=num_classes, pretrained=False)
    elif model_name == 'BiSeNet_res101':
        return  BiSeNet(backbone='resnet101', n_classes=num_classes, pretrained=False)

    elif model_name == 'ENet':
        return ENet(classes=num_classes)

    ## GALDNet
    elif model_name == 'GALD_res50':
        return GALD_res50(num_classes=num_classes)
