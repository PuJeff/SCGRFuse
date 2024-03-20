# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv3Lrelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3Lrelu, self).__init__()
        self.conv3d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = F.leaky_relu(self.conv3d(x), negative_slope=0.2)
        return x

class Conv1Lrelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(Conv1Lrelu, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = F.leaky_relu(self.conv1d(x), negative_slope=0.2)
        return x

class laplacian(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(laplacian, self).__init__()
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.convl = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convl.weight.data.copy_(torch.from_numpy(kernel))

    def forward(self, x):
        x = self.convl(x)
        x = torch.abs(x)
        return x

class RXDB(nn.Module):
   def __init__(self, in_channels, out_channels):
       super(RXDB, self).__init__()
       self.laplaconv = laplacian(in_channels)
       self.conv1 = Conv1Lrelu(in_channels, in_channels)
       self.conv2 = Conv3Lrelu(in_channels, in_channels)
       self.conv3 = Conv3Lrelu(2*in_channels, in_channels)
       self.conv4 = Conv3Lrelu(3*in_channels, in_channels)
       self.conv5 = Conv1Lrelu(10*in_channels, out_channels)
       self.conv6 = Conv1Lrelu(in_channels, out_channels)
       self.sobelconv = Sobelxy(in_channels)

   def forward(self, x):
       x1 = self.laplaconv(x)
       x1 = x + x1
       x11 = self.conv1(x1)
       x12 = torch.cat((x11, self.conv2(x11)), dim=1)
       x13 = torch.cat((x12, self.conv3(x12)), dim=1)
       x14 = torch.cat((x13, self.conv4(x13)), dim=1)
       x15 = self.conv5(torch.cat([x11, x12, x13, x14], dim=1))
       x2 = self.sobelconv(x)
       x2 = self.conv1(x2)
       x2 = self.conv2(x2)
       x2 = self.conv6(x2)
       return F.leaky_relu(x + x2 + x15, negative_slope=0.1)

class MultiScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScale, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, 1, 2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, 7, 1, 3)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv3(x))
        x3 = F.leaky_relu(self.conv5(x))
        x4 = F.leaky_relu(self.conv7(x))
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        return x_cat

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        # self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x1 = torch.tanh(self.conv(x))/2+0.5
        # x1 = self.sigmoid_atten(self.conv(x))
        out = torch.mul(x, x1)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = Conv1Lrelu(in_channels, 2)
        self.conv_atten = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        # self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.mean(x1, 1, True)
        x3 = torch.max(x1, 1, True)[0]
        x4 = torch.cat([x2, x3], dim=1)
        x4 = torch.tanh(self.conv_atten(x4))/2+0.5
        # x4 = self.sigmoid_atten(self.conv_atten(x4))
        out = x*x4
        return out

class PFB(nn.Module):
    def __init__(self):
        super(PFB, self).__init__()
        self.conv = Conv1Lrelu(96*2, 96)

    def forward(self, saf, caf):
        caf = F.avg_pool2d(caf, kernel_size=3, stride=1, padding=1)
        boundary = F.pad(caf, [1, 1, 1, 1], mode='reflect')
        out = torch.cat([saf, caf], dim=1)
        out = self.conv(out)
        return out


# Fusion最好的
class FusionNet(nn.Module):
    def __init__(self, output):
        super(FusionNet, self).__init__()
        vis_ch = [16, 16, 16]
        inf_ch = [16, 16, 16]
        output=1
        self.xconv31 = Conv3Lrelu(2, 128)
        self.xconv32 = Conv3Lrelu(128, 96)

        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1 = RXDB(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = RXDB(vis_ch[1], vis_ch[2])

        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = RXDB(inf_ch[0], inf_ch[1])
        self.inf_rgbd2 = RXDB(inf_ch[1], inf_ch[2])

        self.multi = MultiScale(vis_ch[2]*3+inf_ch[2]*3, vis_ch[2]*3+inf_ch[2]*3)
        self.conv1l = Conv1Lrelu((vis_ch[2]*3+inf_ch[2]*3)*4, vis_ch[2]*3+inf_ch[2]*3)
        self.spa = SpatialAttention(vis_ch[2]*3+inf_ch[2]*3, vis_ch[2]*3+inf_ch[2]*3)
        self.cpa = ChannelAttention(vis_ch[2]*3+inf_ch[2]*3, vis_ch[2]*3+inf_ch[2]*3)
        self.conv3R = Conv3Lrelu(vis_ch[2]*3+inf_ch[2]*3, vis_ch[2]*3+inf_ch[2]*3)
        self.pfb = PFB()

        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2]*3+inf_ch[2]*3, vis_ch[1]*2+vis_ch[1]*2)
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]*2+inf_ch[1]*2, vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        x_s = torch.cat((x_vis_origin, x_inf_origin), dim=1)
        x_s = self.xconv31(x_s)
        x_s = self.xconv32(x_s)
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        x_vis_p2=torch.cat((x_vis_p, x_vis_p1, x_vis_p2), dim=1)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        x_inf_p2 = torch.cat((x_inf_p, x_inf_p1, x_inf_p2), dim=1)

        x0 = torch.cat((x_vis_p2,x_inf_p2),dim=1)
        x0 = self.multi(x0)
        x0 = self.conv1l(x0)
        H, W = x0.size()[2:]
        x1 = self.spa(x0)
        x1 = self.conv3R(x1)
        x2 = self.cpa(x0)
        x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=True)
        x = self.pfb(x1, x2)
        x = x + x_s
        # x = x0 + x_s
        # decode
        x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        return x






def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2,4,480,640).astype(np.float32))
    model = FusionNet(output=1)
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (2,1,480,640), 'output shape (2,1,480,640) is expected!'
    print('test ok!')

if __name__ == '__main__':
    unit_test()


