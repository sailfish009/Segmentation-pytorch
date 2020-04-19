import torch.nn as nn
from tools.utils import netParams
from torchsummary import summary
import torchvision.models as models
import torch.nn.functional as F
import torch


__all__ = ['U_Net', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
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
    def __init__(self, inplanes, planes, bilinear=False):
        super(Upsample, self).__init__()
        self.bilinear = bilinear

        self.up = nn.ConvTranspose2d(inplanes, planes, 2, stride=2)

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
                x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
            else:
                x1 = self.up(x1)

            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            # for padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            # x = self.conv(x)
            x = self.conv3(x)
            x = self.bn3(x)
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out += identity

            out = self.relu(out)
        else:
            if self.bilinear:
                x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
            else:
                x1 = self.up(x1)
            identity = x1

            out = self.conv1(x1)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out += identity

            out = self.relu(out)
        return out



class U_Net(nn.Module):

    def __init__(self, block, layers, classes=3, zero_init_residual=False):
        super(U_Net, self).__init__()
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
            self.Upsample1 = Upsample(512, 256)
            self.Upsample2 = Upsample(256, 128)
            self.Upsample3 = Upsample(128, 64)
            self.Upsample4 = Upsample(64, 64)
            self.Upsample5 = Upsample(64, classes)
        elif block.__name__ == 'Bottleneck':
            self.Upsample1 = Upsample(2048, 1024)
            self.Upsample2 = Upsample(1024, 512)
            self.Upsample3 = Upsample(512, 256)
            self.Upsample4 = Upsample(256, 64)
            self.Upsample5 = Upsample(64, classes)

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
        Up1 = self.Upsample1(Down4, Down3)
        Up2 = self.Upsample2(Up1, Down2)
        Up3 = self.Upsample3(Up2, Down1)
        Up4 = self.Upsample4(Up3)
        Up5 = self.Upsample5(Up4)
        return Up5


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        resnet18 = models.resnet18(pretrained=True)  # 加载官方模型
        cnn = U_Net(BasicBlock, [2, 2, 2, 2], **kwargs)
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
        cnn = U_Net(BasicBlock, [3, 4, 6, 3], **kwargs)
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
        cnn = U_Net(Bottleneck, [3, 4, 6, 3], **kwargs)
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
        cnn = U_Net(Bottleneck, [3, 4, 23, 3], **kwargs)
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
        cnn = U_Net(Bottleneck, [3, 8, 36, 3], **kwargs)
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


def UNet_res(backbone='resnet18', pretrained=True, classes=3):
    if backbone == 'resnet18':
        model = resnet18(pretrained=pretrained, classes=classes)
    elif backbone == 'resnet34':
        model = resnet34(pretrained=pretrained, classes=classes)
    elif backbone == 'resnet50':
        model = resnet50(pretrained=pretrained, classes=classes)
    elif backbone == 'resnet101':
        model = resnet101(pretrained=pretrained, classes=classes)
    elif backbone == 'resnet152':
        model = resnet152(pretrained=pretrained, classes=classes)
    return model


"""print layers and params of network"""
if __name__ == '__main__':
    model = UNet_res(backbone='resnet50', pretrained=True, classes=2)
    summary(model, (3,512,512), device='cpu')

    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))