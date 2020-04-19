import torch
from torch import nn
import torch.nn.functional as F
from tools.utils import netParams
from torchsummary import summary
import model.resnet_backbone as models
# import sys
# sys.path.append('/media/ding/Study/graduate/Segmentation_Torch/model')
# print(sys.path)



class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True):
        super(PSPNet, self).__init__()
        assert layers in [18, 34, 50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        models.BatchNorm = BatchNorm

        if layers == 18:
            resnet = models.resnet18(pretrained=pretrained)
        elif layers == 34:
            resnet = models.resnet34(pretrained=pretrained)
        elif layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        if layers >= 50:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

            fea_dim = 2048
            if use_ppm:
                self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
                fea_dim *= 2
            self.cls = nn.Sequential(
                nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
                BatchNorm(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(512, classes, kernel_size=1)
            )
        else:
            fea_dim = 512
            if use_ppm:
                self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins, BatchNorm)
                fea_dim *= 2
            self.cls = nn.Sequential(
                nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
                BatchNorm(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(512, classes, kernel_size=1)
            )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        x_size = x.size()
        # print('x_size is ', x_size)
        # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        # h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        # w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        h, w = x_size[2], x_size[3]

        x = self.layer0(x)
        # print('layer0 x size is ', x.size())
        x = self.layer1(x)
        # print('layer1 x size is ', x.size())
        x = self.layer2(x)
        # print('layer2 x size is ', x.size())
        x_tmp = self.layer3(x)
        # print('x_temp size is ', x_tmp.size())
        x = self.layer4(x_tmp)
        # print('layer4 x size is ', x.size())

        if self.use_ppm:
            x = self.ppm(x)  # [4, 3, 64, 64]
        x = self.cls(x)
        # print('ppm size is ', x.size())
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        # print('x size is ', x.size())
        # print('self.training is', self.training)
        if self.training:
            aux = self.aux(x_tmp) # [4, 3, 64, 64]

            if self.zoom_factor != 1:
                # print('h & w', h, w)
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            # return x.max(1)[1], main_loss, aux_loss
            return x, aux_loss
        else:
            return x



if __name__ == '__main__':
    # cuda
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # input = torch.Tensor(1,3,512,512).cuda()
    # model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=3, zoom_factor=8, use_ppm=True, pretrained=True).cuda()
    # model.eval()
    # print(model)
    # output = model(input)
    # print('PSPNet', output.size())
    # summary(model, (3, 512, 512))

    # cpu
    input = torch.Tensor(1, 3, 512, 512)
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=3, zoom_factor=8, use_ppm=True,
                   pretrained=True)
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output.size())
    summary(model, (3, 512, 512), device='cpu')

    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))
