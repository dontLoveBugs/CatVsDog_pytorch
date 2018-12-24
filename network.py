# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/24 20:08
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""
import math

import torch
import torch.nn as nn
import torchvision


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class CatDogClassifier(nn.Module):

    def __init__(self, layers=101, pretrained=True):
        super(CatDogClassifier, self).__init__()

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(
                'Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']

        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(2048, 2)

        self.avgpool.apply(weights_init)
        self.classifier.apply(weights_init)

        del pretrained_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # print(x.size())

        x = self.avgpool(x)

        # print(x.size())
        x = x.view(-1, 2048)
        x = self.classifier(x)
        return x


if __name__ == '__main__':

    img = torch.randn(10, 3, 112, 112).float().cuda()

    model = CatDogClassifier().cuda()

    label = model(img)

    print(label.size())

