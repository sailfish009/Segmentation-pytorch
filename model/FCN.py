import torch.nn as nn
from tools.utils import netParams
from torchsummary import summary
import torchvision.models as models
import torch.nn.functional as F
import torch


__all__ = ['FCN', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Upsample(nn.Module):
    def __init__(self, inplanes, planes, bilinear=False, scale_factor=2):
        super(Upsample, self).__init__()
        self.bilinear = bilinear
        self.scale_factor = scale_factor
        self.inplanes = inplanes
        self.planes = planes


        self.conv1 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(inplanes, planes)
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x1, x2=None):
        if x2 is not None:
            if self.bilinear:
                x1 = F.interpolate(x1, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
                x1 = self.conv3(x1)  # 1×1卷积改通道数
                x1 = self.bn3(x1)
            else:
                if self.scale_factor == 2:
                    # x1 = self.up(x1)
                    x1 = nn.ConvTranspose2d(self.inplanes, self.planes, 2, stride=2)(x1)
                elif self.scale_factor == 8:
                    # x1 = self.up8(x1)
                    x1 = nn.ConvTranspose2d(self.inplanes, self.planes, 8, stride=8)(x1)
                elif self.scale_factor == 16:
                    # x1 = self.up16(x1)
                    x1 = nn.ConvTranspose2d(self.inplanes, self.planes, 16, stride=16)(x1)
                elif self.scale_factor == 32:
                    # x1 = self.up32(x1)
                    x1 = nn.ConvTranspose2d(self.inplanes, self.planes, 32, stride=32)(x1)
            # for padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = x1 + x2
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out += identity

            out = self.relu(out)
            return out
        else:
            if self.bilinear:
                x1 = F.interpolate(x1, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
                x1 = self.conv3(x1)  # 1×1卷积改通道数
                x1 = self.bn3(x1)
            else:
                if self.scale_factor == 2:
                    x1 = nn.ConvTranspose2d(self.inplanes, self.planes, 2, stride=2)(x1)
                elif self.scale_factor == 8:
                    x1 = nn.ConvTranspose2d(self.inplanes, self.planes, 8, stride=8)(x1)
                elif self.scale_factor == 16:
                    x1 = nn.ConvTranspose2d(self.inplanes, self.planes, 16, stride=16)(x1)
                elif self.scale_factor == 32:
                    x1 = nn.ConvTranspose2d(self.inplanes, self.planes, 32, stride=32)(x1)

            identity = x1

            out = self.conv1(x1)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out += identity

            out = self.relu(out)
            return out



class FCN(nn.Module):

    def __init__(self, block, layers, classes=3, zero_init_residual=False, scale=8):
        super(FCN, self).__init__()
        self.scale = scale
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block.__name__ == 'BasicBlock':
            if self.scale == 32:
                self.Upsample32 = Upsample(inplanes=512, planes=classes, bilinear=True, scale_factor=32)
            if self.scale == 16:
                self.Upsample2 = Upsample(inplanes=512, planes=256, bilinear=True, scale_factor=2)
                self.Upsample16 = Upsample(inplanes=256, planes=classes, bilinear=True, scale_factor=16)
            if self.scale == 8:
                self.Upsample2_1 = Upsample(inplanes=512, planes=256, bilinear=False, scale_factor=2)
                self.Upsample2_2 = Upsample(inplanes=256, planes=128, bilinear=False, scale_factor=2)
                self.Upsample8 = Upsample(inplanes=128, planes=classes, bilinear=False, scale_factor=8)
        elif block.__name__ == 'Bottleneck':
            if self.scale == 32:
                self.Upsample32 = Upsample(inplanes=2048, planes=classes, bilinear=True, scale_factor=32)
            if self.scale == 16:
                self.Upsample2 = Upsample(inplanes=2048, planes=1024, bilinear=False, scale_factor=2)
                self.Upsample16 = Upsample(inplanes=1024, planes=classes, bilinear=False, scale_factor=16)
            if self.scale == 8:
                self.Upsample2_1 = Upsample(inplanes=2048, planes=1024, bilinear=True, scale_factor=2)
                self.Upsample2_2 = Upsample(inplanes=1024, planes=512, bilinear=True, scale_factor=2)
                self.Upsample8 = Upsample(inplanes=512, planes=classes, bilinear=True, scale_factor=8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)  #256 256 64 64
        x2 = self.maxpool(x1)    #128 128 64 64

        Down1 = self.layer1(x2)  #128 128 64 256
        Down2 = self.layer2(Down1)  #64 64 128 512
        Down3 = self.layer3(Down2)  #32 32 256 1024
        Down4 = self.layer4(Down3)  #16 16 512 2048
        # print(x1.size(), x2.size(), Down1.size(), Down2.size(), Down3.size(), Down4.size())
        if self.scale == 32:
            output = self.Upsample32(Down4)
            return output
        if self.scale == 16:
            temp = self.Upsample2(Down4, Down3)
            output = self.Upsample16(temp)
            return output
        if self.scale == 8:
            temp = self.Upsample2_1(Down4, Down3)
            temp = self.Upsample2_2(temp, Down2)
            output = self.Upsample8(temp)
            return output



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        resnet18 = models.resnet18(pretrained=True)  # 加载官方模型
        cnn = FCN(BasicBlock, [2, 2, 2, 2], **kwargs)
        # 读取参数
        pretrained_dict = resnet18.state_dict()  # 加载官方模型
        model_dict = cnn.state_dict()
        # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        # 加载我们真正需要的state_dict
        cnn.load_state_dict(model_dict)

        return cnn


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        resnet34 = models.resnet34(pretrained=True)  # 加载官方模型
        cnn = FCN(BasicBlock, [3, 4, 6, 3], **kwargs)
        # 读取参数
        pretrained_dict = resnet34.state_dict()  # 加载官方模型
        model_dict = cnn.state_dict()
        # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        # 加载我们真正需要的state_dict
        cnn.load_state_dict(model_dict)
        return cnn


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        resnet50 = models.resnet50(pretrained=True)  # 加载官方模型
        cnn = FCN(Bottleneck, [3, 4, 6, 3], **kwargs)
        # 读取参数
        pretrained_dict = resnet50.state_dict()  # 加载官方模型
        model_dict = cnn.state_dict()
        # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        # 加载我们真正需要的state_dict
        cnn.load_state_dict(model_dict)

        return cnn


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        resnet101 = models.resnet101(pretrained=True)  # 加载官方模型
        cnn = FCN(Bottleneck, [3, 4, 23, 3], **kwargs)
        # 读取参数
        pretrained_dict = resnet101.state_dict()  # 加载官方模型
        model_dict = cnn.state_dict()
        # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        # 加载我们真正需要的state_dict
        cnn.load_state_dict(model_dict)

        return cnn


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        resnet152 = models.resnet152(pretrained=True)  # 加载官方模型
        cnn = FCN(Bottleneck, [3, 8, 36, 3], **kwargs)
        # 读取参数
        pretrained_dict = resnet152.state_dict()  # 加载官方模型
        model_dict = cnn.state_dict()
        # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        # 加载我们真正需要的state_dict
        cnn.load_state_dict(model_dict)

        return cnn


def FCN_res(backbone='resnet18', pretrained=True, classes=3, scale=32):
    if backbone == 'resnet18':
        model = resnet18(pretrained=pretrained, classes=classes, scale=scale)
    elif backbone == 'resnet34':
        model = resnet34(pretrained=pretrained, classes=classes, scale=scale)
    elif backbone == 'resnet50':
        model = resnet50(pretrained=pretrained, classes=classes, scale=scale)
    elif backbone == 'resnet101':
        model = resnet101(pretrained=pretrained, classes=classes, scale=scale)
    elif backbone == 'resnet152':
        model = resnet152(pretrained=pretrained, classes=classes, scale=scale)
    return model


"""print layers and params of network"""
if __name__ == '__main__':
    # cuda
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # input = torch.Tensor(1,3,512,512).cuda()
    # model = FCN_res(backbone='resnet18', classes=3, pretrained=True, scale=16).cuda()  # scale=[8,16,32]
    # model.eval()
    # print(model)
    # output = model(input)
    # print('FCN_res', output.size())
    # summary(model, (3, 512, 512))

    # cpu
    input = torch.Tensor(1, 3, 512, 512)
    model = FCN_res(backbone='resnet50', classes=3, pretrained=True, scale=8)  # scale=[8,16,32]
    model.eval()
    # print(model)
    output = model(input)
    print('FCN_res', output.size())
    summary(model, (3, 512, 512), device='cpu')

    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))
