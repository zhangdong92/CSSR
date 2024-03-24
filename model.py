import torch
import torch.nn as nn
import torch.nn.functional as F


class downSample(nn.Module):
    """u-net down sample layer"""


    def __init__(self, inputChannelNum, outputChannelNum, filterSize):
        super(downSample, self).__init__()
        self.c1 = nn.Conv2d(inputChannelNum,  outputChannelNum, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.c2 = nn.Conv2d(outputChannelNum, outputChannelNum, filterSize, stride=1, padding=int((filterSize - 1) / 2))
           
    def forward(self, x):
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.c1(x), negative_slope = 0.05)
        x = F.leaky_relu(self.c2(x), negative_slope = 0.05)
        return x
    
class upSample(nn.Module):
    """u-net up sample layer"""

    def __init__(self, inputChannelNum, outputChannelNum):
        super(upSample, self).__init__()
        self.c1 = nn.Conv2d(inputChannelNum,  outputChannelNum, 3, stride=1, padding=1)
        self.c2 = nn.Conv2d(2 * outputChannelNum, outputChannelNum, 3, stride=1, padding=1)
           
    def forward(self, x, skpCn):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = F.leaky_relu(self.c1(x), negative_slope = 0.05)
        x = F.leaky_relu(self.c2(torch.cat((x, skpCn), 1)), negative_slope = 0.05)
        return x


class UNet1(nn.Module):
    """
    u-net1, output deformation field. input 6 channels
    without attention
    """

    def __init__(self, inputChannelNum, outputChannelNum):
        super(UNet1, self).__init__()
        self.c1 = nn.Conv2d(inputChannelNum, 40, 7, stride=1, padding=3)
        self.c2 = nn.Conv2d(40, 40, 7, stride=1, padding=3)

        self.downSample1 = downSample(40, 80, 5)
        self.downSample2 = downSample(80, 160, 3)
        self.downSample3 = downSample(160, 320, 3)
        self.downSample4 = downSample(320, 640, 3)
        self.downSample5 = downSample(640, 640, 3)
        self.upSample1 = upSample(640, 640)
        self.upSample2 = upSample(640, 320)
        self.upSample3 = upSample(320, 160)
        self.upSample4 = upSample(160, 80)
        self.upSample5 = upSample(80, 40)
        self.c3 = nn.Conv2d(40, outputChannelNum, 3, stride=1, padding=1)


    def forward(self, x):
        x = F.leaky_relu(self.c1(x), negative_slope=0.05)
        s1 = F.leaky_relu(self.c2(x), negative_slope=0.05)
        s2 = self.downSample1(s1)
        s3 = self.downSample2(s2)
        s4 = self.downSample3(s3)
        s5 = self.downSample4(s4)
        x = self.downSample5(s5)
        x = self.upSample1(x, s5)
        x = self.upSample2(x, s4)
        x = self.upSample3(x, s3)
        x = self.upSample4(x, s2)
        x = self.upSample5(x, s1)
        x = F.leaky_relu(self.c3(x), negative_slope=0.05)
        return x


class UNet2(nn.Module):
    """
    u-net2, output deformation field refinement. input 6 channels
    without attention
    """


    def __init__(self, inputChannelNum, outputChannelNum):
        super(UNet2, self).__init__()

        # attention block
        self.cbam0 = CBAM(channel=inputChannelNum,ratio=4)

        self.c1 = nn.Conv2d(inputChannelNum, 40, 7, stride=1, padding=3)
        self.c2 = nn.Conv2d(40, 40, 7, stride=1, padding=3)
        self.downSample1 = downSample(40, 80, 5)
        self.downSample2 = downSample(80, 160, 3)
        self.downSample3 = downSample(160, 320, 3)
        self.downSample4 = downSample(320, 640, 3)
        self.downSample5 = downSample(640, 640, 3)
        self.upSample1   = upSample(640, 640)
        self.upSample2   = upSample(640, 320)
        self.upSample3   = upSample(320, 160)
        self.upSample4   = upSample(160, 80)
        self.upSample5   = upSample(80, 40)
        self.c3 = nn.Conv2d(40, outputChannelNum, 3, stride=1, padding=1)



    def forward(self, x):
        x = self.cbam0(x) + x
        x  = F.leaky_relu(self.c1(x), negative_slope = 0.05)
        # x = self.cbam1(x) + x

        s1 = F.leaky_relu(self.c2(x), negative_slope = 0.05)
        # # replace 7*7 with three 3*3conv
        # x = F.leaky_relu(self.c2_1(x), negative_slope = 0.05)
        # x = F.leaky_relu(self.c2_2(x), negative_slope = 0.05)
        # s1 = F.leaky_relu(self.c2_3(x), negative_slope = 0.05)

        # s1 = self.cbam2(s1) + s1
        s2 = self.downSample1(s1)
        # s2 = self.cbam_d1(s2) + s2
        s3 = self.downSample2(s2)
        # s3 = self.cbam_d2(s3) + s3
        s4 = self.downSample3(s3)
        # s4 = self.cbam_d3(s4) + s4
        s5 = self.downSample4(s4)
        # s5 = self.cbam_d4(s5) + s5
        x  = self.downSample5(s5)
        x  = self.upSample1(x, s5)
        x  = self.upSample2(x, s4)
        x  = self.upSample3(x, s3)
        x  = self.upSample4(x, s2)
        x  = self.upSample5(x, s1)
        x  = F.leaky_relu(self.c3(x), negative_slope = 0.05)
        return x


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    """attention block. CBAM, Convolutional Block Attention Module """
    def __init__(self, channel,ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel,ratio)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out

        # out = self.spatial_attention(x) * x
        return out
