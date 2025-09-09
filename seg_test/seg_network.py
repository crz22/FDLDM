import torch
import torch.nn as nn

class CONV3D_BLOCK(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,padding=1):
        super(CONV3D_BLOCK, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding),
                                  nn.BatchNorm3d(out_channel),
                                  nn.ReLU(inplace=True))
    def forward(self,x):
        return self.conv(x)

class RES_BLOCK(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,padding=1):
        super(RES_BLOCK, self).__init__()
        self.conv1 = CONV3D_BLOCK(in_channel,out_channel,kernel_size,stride,padding)
        self.conv2 = nn.Sequential(nn.Conv3d(out_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding),
                                   nn.BatchNorm3d(out_channel))

        self.short_cut = in_channel != out_channel
        if self.short_cut:
            self.short_conv = nn.Conv3d(in_channel,out_channel,kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.short_cut:
            residual = self.short_conv(residual)
        return self.act(x + residual)

class Downsample(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Downsample, self).__init__()
        self.down_layer = nn.Sequential(nn.Conv3d(in_channel,out_channel,kernel_size=3,stride=2,padding=1),
                                        nn.BatchNorm3d(out_channel),
                                        nn.ReLU(inplace=True))
        self.res_layer = RES_BLOCK(out_channel,out_channel)
    def forward(self,x):
        x = self.down_layer(x)
        x = self.res_layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Upsample, self).__init__()
        self.up_layer = nn.Upsample(scale_factor=2)
        self.conv = CONV3D_BLOCK(in_channel+out_channel,out_channel,kernel_size=3,stride=1,padding=1)
        self.res_layer = RES_BLOCK(out_channel,out_channel)

    def forward(self,x,x_cat):
        x = self.up_layer(x)
        x = self.conv(torch.cat([x,x_cat],dim=1))
        x = self.res_layer(x)
        return x

class UNet3D_RES(nn.Module):
    def __init__(self,in_channel,out_channel,mid_channel=64):
        super(UNet3D_RES, self).__init__()
        #[1,32,32,32]
        self.input_layer = CONV3D_BLOCK(in_channel,mid_channel)
        #[64,32,32,32]
        self.down_layer1 = Downsample(mid_channel,mid_channel*2)
        #[128,16,16,16]
        self.down_layer2 = Downsample(mid_channel*2,mid_channel*4)
        #[256,8,8,8]
        self.down_layer3 = Downsample(mid_channel*4,mid_channel*8)
        #[512,4,4,4]
        self.res_layer = RES_BLOCK(mid_channel*8,mid_channel*8)
        #[512,4,4,4]
        self.up_layer1 = Upsample(mid_channel*8,mid_channel*4)
        #[256,8,8,8]
        self.up_layer2 = Upsample(mid_channel*4,mid_channel*2)
        #[128,16,16,16]
        self.up_layer3 = Upsample(mid_channel*2,mid_channel)
        #[64,32,32,32]
        self.out_layer = nn.Conv3d(mid_channel,out_channel,kernel_size=1)

    def forward(self,x):
        x = self.input_layer(x)
        x_1 = self.down_layer1(x)
        x_2 = self.down_layer2(x_1)
        x_3 = self.down_layer3(x_2)

        x_3 = self.res_layer(x_3)

        x_4 = self.up_layer1(x_3,x_2)
        x_5 = self.up_layer2(x_4,x_1)
        x_6 = self.up_layer3(x_5,x)

        out = self.out_layer(x_6)
        return out





