from torch import nn
from torchvision import models

from utils import *

vgg16 = models.vgg16(pretrained=True)


class fcn8s(nn.Module):
    def __init__(self, num_classes):
        super(fcn8s, self).__init__()

        self.stage1 = nn.Sequential(
            *list(vgg16.features[:-14])
        )
        self.stage2 = nn.Sequential(
            *list(vgg16.features[-14:-7])
        )
        self.stage3 = nn.Sequential(
            *list(vgg16.features[-7:])
        )

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(512, num_classes, 1)
        self.scores3 = nn.Conv2d(256, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel

        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

    def forward(self, x):
        x = self.stage1(x)
        s1 = x

        x = self.stage2(x)
        s2 = x

        x = self.stage3(x)

        s3 = self.scores1(x)
        s3 = self.upsample_2x(s3)

        s2 = self.scores2(s2)
        s2 = s2 + s3

        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2

        s = self.upsample_8x(s)
        return s
