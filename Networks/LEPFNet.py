from tkinter import Y
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
from PIL import Image



# 普通的残差块
class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)

        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


# 光线增强块
class CPM_ResidualBlock(nn.Module):
    def __init__(self, cpm_channal_in=32, cpm_channal_out=32, res_channel_num=64, dilation=1):
        super(CPM_ResidualBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.res_conv = ResidualBlock(res_channel_num, dilation)

        self.cpm_conv_1 = nn.Conv2d(cpm_channal_in, cpm_channal_in, 3, 1, 1, groups=cpm_channal_in)
        self.cpm_conv_2 = nn.Conv2d(cpm_channal_in, cpm_channal_out, 1, 1)

    def forward(self, conv_x, conv_cpm_x1=None, conv_cpm_x2=None):
        if conv_cpm_x2 is None:
            res_y = self.res_conv(conv_x)
            cpm_y = self.relu(self.cpm_conv_2(self.cpm_conv_1(conv_cpm_x1)))
            return res_y, cpm_y
        else:
            res_y = self.res_conv(conv_x)
            cpm_y = self.relu(self.cpm_conv_2(self.cpm_conv_1(torch.cat([conv_cpm_x1, conv_cpm_x2], 1))))
            return res_y, cpm_y


# 编码器
class Encoder(nn.Module):
    def __init__(self, channal_in):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(channal_in, 64, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)

        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)

        self.conv3 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y = F.relu(self.norm3(self.conv3(y)))

        return y


# 自适应特征融合子网络
class FF_SubNet(nn.Module):
    def __init__(self):
        super(FF_SubNet, self).__init__()

        self.AAP2d = nn.AdaptiveAvgPool2d(1)

        self.ReLu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(64 * 3, 64 // 16, 1, padding=0)
        self.conv2 = nn.Conv2d(64 // 16, 64 * 3, 1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(64 * 3, 64 // 8, 1, padding=0, bias=True)
        self.conv4 = nn.Conv2d(64 // 8, 64 * 3, 1, padding=0, bias=True)

    def forward(self, y1, y2, y3):
        y = self.AAP2d(torch.cat([y1, y2, y3], dim=1))

        y = self.ReLu(self.conv1(y))
        y = self.ReLu(self.conv2(y))
        y = self.ReLu(self.conv3(y))
        y = self.ReLu(self.conv4(y))
        
        y = y.view(-1, 3, 64)[:, :, :, None, None]
        y_hat = y[:, 0, ::] * y1 + y[:, 1, ::] * y2 + y[:, 2, ::] * y3

        return y_hat


# 中间块
class MiddleBlock(nn.Module):
    def __init__(self):
        super(MiddleBlock, self).__init__()

        # Have CPM
        self.res1 = CPM_ResidualBlock(3, 32, 64, 2)
        self.res2 = CPM_ResidualBlock(32, 32, 64, 2)
        self.res3 = CPM_ResidualBlock(32, 32, 64, 2)
        self.res4 = CPM_ResidualBlock(32, 32, 64, 4)
        self.res5 = CPM_ResidualBlock(64, 32, 64, 4)
        self.res6 = CPM_ResidualBlock(64, 32, 64, 4)
        self.res7 = CPM_ResidualBlock(64, 3, 64, 1)

        # FF Net
        self.FFB = FF_SubNet()

    def forward(self, res_x, cpm_x):
        # CPM残差块
        y, x1 = self.res1(res_x, cpm_x)
        y, x2 = self.res2(y, x1)
        y, x3 = self.res3(y, x2)
        y2, x4 = self.res4(y, x3)
        y, x5 = self.res5(y2, x3, x4)
        y, x6 = self.res6(y, x2, x5)
        y3, cpm_y = self.res7(y, x1, x6)

        # 特征融合块
        res_y = self.FFB(res_x, y2, y3)
        return res_y, cpm_y

# 解码器
class Decoder(nn.Module):
    def __init__(self, channal_out):
        super(Decoder, self).__init__()

        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.norm4 = nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm5 = nn.InstanceNorm2d(64, affine=True)

        self.deconv1 = nn.Conv2d(64, channal_out, 1)

    def forward(self, x):
        y = F.relu(self.norm4(self.deconv3(x)))
        y = F.relu(self.norm5(self.deconv2(y)))
        cleanImage = self.deconv1(y)

        return cleanImage


# XXX模型
class dehaze_net(nn.Module):
    def __init__(self, channal_in=3, channal_out=3):
        super(dehaze_net, self).__init__()

        # 编码器
        self.enCoder = Encoder(channal_in)

        # 中间块
        self.midBlock = MiddleBlock()

        # 解码器
        self.deCoder = Decoder(channal_out)

    def get_cleanImage_and_cpm(self, x, get_cpm=True):
        # 编码器
        y1 = self.enCoder(x)

        # 中间块
        y, cpm = self.midBlock(y1, x)

        # fileName = "/home/jqyan/YiJianWang/torch-dehazing/1.jpg"
        # model = nn.Conv2d(64, 1, 1)
        # model = model.cuda()
        # torchvision.utils.save_image(cpm, fileName)

        # torch.set_printoptions(profile="full")
        # print(cpm)

        # 解码器
        cleanImage = self.deCoder(y)

        if get_cpm is True:
            return cleanImage, cpm
        else:
            return cleanImage

    def forward(self, x, get_cpm=True):
        return self.get_cleanImage_and_cpm(x, get_cpm)