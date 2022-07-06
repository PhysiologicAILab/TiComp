import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
from .modules.CBAM import CBAM, ChannelGate, ChannelPool

relu_slope = 0.2       #Default value 0.01
norm_layer_1 = nn.BatchNorm2d
norm_layer_2 = nn.BatchNorm2d

class InitialConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=1, padding_mode='reflect', bias=False),
            norm_layer_1(out_channels),
            nn.LeakyReLU(negative_slope=relu_slope),
        )

    def forward(self, x):
        x = x + self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()

        n_ch1 = in_channels
        n_ch2 = in_channels

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(n_ch1, n_ch2, kernel_size=3, stride=1, dilation=1, padding=1, padding_mode='reflect', bias=False)),
            norm_layer_1(n_ch2),
            nn.LeakyReLU(negative_slope=relu_slope),
            spectral_norm(nn.Conv2d(n_ch2, n_ch2, kernel_size=3, stride=1, dilation=1, padding=1, padding_mode='reflect', bias=False)),
            norm_layer_1(n_ch2),
            nn.LeakyReLU(negative_slope=relu_slope),
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(n_ch1, n_ch2, kernel_size=3, stride=1, dilation=2, padding=2, padding_mode='reflect', bias=False)),
            norm_layer_1(n_ch2),
            nn.LeakyReLU(negative_slope=relu_slope),
            spectral_norm(nn.Conv2d(n_ch2, n_ch2, kernel_size=3, stride=1, dilation=2, padding=2, padding_mode='reflect', bias=False)),
            norm_layer_1(n_ch2),
            nn.LeakyReLU(negative_slope=relu_slope),
        )

        # self.conv_merge = nn.Sequential(
        #     nn.Conv2d(3*n_ch2, n_ch2, kernel_size=1),
        #     norm_layer_1(n_ch2),
        #     nn.LeakyReLU(negative_slope=relu_slope),
        # )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        # x = torch.cat([x, x1, x2], dim=1)
        # x = self.conv_merge(x)
        x = x + x1 + x2
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        n_ch1 = in_channels
        n_ch2 = in_channels # in_channels // 2
        n_ch3 = out_channels #out_channels // 3
        self.conv_1a = nn.Sequential(
            spectral_norm(nn.Conv2d(n_ch1, n_ch2, kernel_size=3, stride=1, dilation=1, padding=1, padding_mode='reflect', bias=False)),
            norm_layer_2(n_ch2),
            nn.LeakyReLU(negative_slope=relu_slope),
        )
        self.conv_1b = nn.Sequential(
            spectral_norm(nn.Conv2d(n_ch1, n_ch2, kernel_size=3, stride=1, dilation=2, padding=2, padding_mode='reflect', bias=False)),
            norm_layer_2(n_ch2),
            nn.LeakyReLU(negative_slope=relu_slope),
        )
        self.conv_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(n_ch2, n_ch3, kernel_size=3, stride=2, dilation=1, padding=1, padding_mode='reflect', bias=False)),
            norm_layer_2(n_ch3),
            nn.LeakyReLU(negative_slope=relu_slope),
        )

    def forward(self, x):
        x1a = self.conv_1a(x)
        x1b = self.conv_1b(x)
        x = self.conv_2(x + x1a + x1b)
        return x


class BottleNeck(nn.Module):
    def __init__(self, n_ch1, n_ch2):
        super(BottleNeck, self).__init__()
        in_channels = n_ch1
        out_channels = n_ch2
        self.conv = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            norm_layer_2(out_channels),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)),
            norm_layer_2(out_channels),
            nn.ReLU(),
        )       
        self.cbam = CBAM(gate_channels=out_channels)

    def forward(self, z_enc):
        z_dec = self.conv(z_enc)
        z_dec = self.cbam(z_dec)
        return z_dec


class SkipAttentionGate(nn.Module):
    def __init__(self, layer_channels, reduction_ratio, pool_types=['avg', 'max']):
        super(SkipAttentionGate, self).__init__()
        self.ChannelGate = ChannelGate(layer_channels, reduction_ratio, pool_types=pool_types)
        kernel_size = 7
        self.g_compress = ChannelPool()
        self.x_compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1) // 2),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True) ,
            nn.Sigmoid()
        )

    def forward(self, g, x):
        x = self.ChannelGate(x)
        g_compress = self.g_compress(g)
        x_compress = self.x_compress(x)
        attention = self.spatial(torch.cat([g_compress, x_compress], dim=1))
        return x * attention

class UpConv(nn.Module):
    def __init__(self, n_ch1, n_ch2):
        super(UpConv, self).__init__()
        in_channels = 2*n_ch1
        mid_channels = n_ch1
        out_channels = n_ch2
        reduction_ratio = 12
        self.skip_att_gate = SkipAttentionGate(layer_channels=n_ch1, reduction_ratio=reduction_ratio)
        self.conv = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            norm_layer_2(mid_channels),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)),
            norm_layer_2(out_channels),
            nn.ReLU(),
        )
        self.cbam = CBAM(gate_channels=out_channels)

    def forward(self, z_enc, z_dec):
        # z_enc = self.att_gate(g=z_dec, x=z_enc)
        z_enc = self.skip_att_gate(g=z_dec, x=z_enc)
        z_dec = torch.cat([z_dec, z_enc], dim=1)
        z_dec = self.conv(z_dec)
        z_dec = self.cbam(z_dec)
        return z_dec


class UpConv_Final(nn.Module):
    def __init__(self, n_ch1, n_ch2):
        super(UpConv_Final, self).__init__()
        reduction_ratio = 12
        self.skip_att_gate = SkipAttentionGate(layer_channels=n_ch1, reduction_ratio=reduction_ratio)
        self.conv_final = nn.Sequential(
            nn.Conv2d(2*n_ch1, n_ch1, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            norm_layer_2(n_ch1),
            nn.ReLU(),
            nn.Conv2d(n_ch1, n_ch2, kernel_size=1),
        )

    def forward(self, z_enc, z_dec):
        # z_enc = self.att_gate(g=z_dec, x=z_enc)
        z_enc = self.skip_att_gate(g=z_dec, x=z_enc)
        z_dec = torch.cat([z_dec, z_enc], dim=1)
        z_dec = self.conv_final(z_dec)
        return z_dec

class Generator(nn.Module):
    def __init__(self, n_channels, n_classes, n_features=24, seg_net_out_opt=1):
        super(Generator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_filters = self.n_features * np.array([1, 2, 4, 8, 16])
        self.seg_net_out_opt = seg_net_out_opt

        self.conv1 = InitialConv(self.n_channels, self.n_filters[0])
        
        self.conv_res_1 = ResBlock(self.n_filters[0])
        self.conv_res_2 = ResBlock(self.n_filters[0])
        self.conv_res_3 = ResBlock(self.n_filters[0])

        self.conv_down_1 = DownConv(self.n_filters[0], self.n_filters[1])
        self.conv_down_2 = DownConv(self.n_filters[1], self.n_filters[2])
        self.conv_down_3 = DownConv(self.n_filters[2], self.n_filters[3])
        self.conv_down_4 = DownConv(self.n_filters[3], self.n_filters[4])

        self.conv_up_1 = BottleNeck(n_ch1=self.n_filters[4], n_ch2=self.n_filters[3])
        self.conv_up_2 = UpConv(n_ch1=self.n_filters[3], n_ch2=self.n_filters[2])
        self.conv_up_3 = UpConv(n_ch1=self.n_filters[2], n_ch2=self.n_filters[1])
        self.conv_up_4 = UpConv(n_ch1=self.n_filters[1], n_ch2=self.n_filters[0])
        self.conv_final = UpConv_Final(n_ch1=self.n_filters[0], n_ch2=self.n_classes)


    def forward(self, input_img):

        x1 = self.conv1(input_img)

        x1 = self.conv_res_1(x1)

        x1 = self.conv_res_2(x1)

        x1 = self.conv_res_3(x1)

        x2 = self.conv_down_1(x1)

        x3 = self.conv_down_2(x2)

        x4 = self.conv_down_3(x3)

        z0 = self.conv_down_4(x4)

        z1 = self.conv_up_1(z_enc=z0)
        
        z2 = self.conv_up_2(z_enc=x4, z_dec=z1)

        z3 = self.conv_up_3(z_enc=x3, z_dec=z2)
        
        z4 = self.conv_up_4(z_enc=x2, z_dec=z3)
        
        pred = self.conv_final(z_enc=x1, z_dec=z4)
        
        if self.seg_net_out_opt == 1:
            return pred
        else:
            return pred, z4


if __name__ == '__main__':

    import os
    from torch.utils.tensorboard import SummaryWriter
    from pytorch_model_summary import summary

    n_channels = 1
    n_features = 24
    n_classes = 6
    batch_size = 1
    sim_ht = 256
    sim_wt = 256
    sim_img = torch.zeros((batch_size, n_channels, int(sim_wt), int(sim_ht)))

    logdir = r'runs/tmp/GSNet/Generator'
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    gen = Generator(n_channels, n_features, n_classes=n_classes)
    pred_seg = gen(sim_img)
    print("\npred_seg.shape:", pred_seg.shape)
    pth = os.path.join(logdir, "gen")
    print(summary(gen, sim_img, show_input=True))
    tb_writer = SummaryWriter(logdir)
    with torch.no_grad():
        tb_writer.add_graph(gen, sim_img)
        tb_writer.add_scalar("tmp", 1, 1)
        tb_writer.flush()

    tb_writer.close()
