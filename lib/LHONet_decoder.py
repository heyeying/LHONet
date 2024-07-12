import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class UP(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UP, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Down(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(ch_in, ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, dilation=dilation),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()   # (1,512,22,22)
        y = self.avg_pool(x).view(b, c)     # (1, 512)
        y = self.fc(y).view(b, c, 1, 1)     # (1, 512, 1, 1)
        return x * y.expand_as(x)


class GateFusion(nn.Module):
    def __init__(self, in_planes):
        self.init__ = super(GateFusion, self).__init__()

        self.gate_1 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.gate_2 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        ###
        cat_fea = torch.cat([x1, x2], dim=1)

        ###
        att_vec_1 = self.gate_1(cat_fea)
        att_vec_2 = self.gate_2(cat_fea)

        att_vec_cat = torch.cat([att_vec_1, att_vec_2], dim=1)
        att_vec_soft = self.softmax(att_vec_cat)

        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2

        return x_fusion


class Local_conv(nn.Module):
    def __init__(self, in_channel, out_channel, exp_ratio=1.0):
        super(Local_conv, self).__init__()

        mid_channel = in_channel * exp_ratio

        self.DWConv = ConvBNReLU(mid_channel, mid_channel, kernel_size=3, groups=out_channel // 2)
        self.DWConv3x3 = ConvBNReLU(in_channel // 4, in_channel // 4, kernel_size=3, groups=in_channel // 4)
        self.DWConv5x5 = ConvBNReLU(in_channel // 4, in_channel // 4, kernel_size=5, groups=in_channel // 4)
        self.DWConv7x7 = ConvBNReLU(in_channel // 4, in_channel // 4, kernel_size=7, groups=in_channel // 4)
        self.PWConv1 = ConvBNReLU(in_channel, mid_channel, kernel_size=1)
        self.PWConv2 = ConvBNReLU(mid_channel, out_channel, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channel)
        # MaxPool2d 捕获高频信息
        self.Maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        # self.gelu = nn.GELU()
        # self.proj = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)

        channels = x.size(1)
        channels_per_part = channels // 4
        x1 = x[:, :channels_per_part, :, :]
        x2 = x[:, channels_per_part:2*channels_per_part, :, :]
        x3 = x[:, 2*channels_per_part:3*channels_per_part, :, :]
        x4 = x[:, 3*channels_per_part:, :, :]
        x1 = self.Maxpool(x1)
        x2 = self.DWConv3x3(x2)
        x3 = self.DWConv5x5(x3)
        x4 = self.DWConv7x7(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.PWConv1(x)
        x = x + self.DWConv(x)
        x = self.PWConv2(x)
        x = x + shortcut

        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class dilation_conv(nn.Module):
    def __init__(self, channel, exp_ratio):
        super().__init__()
        self.dila_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel, dilation=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=2, groups=channel, dilation=2),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=3, groups=channel, dilation=3),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.norm = LayerNormChannel(num_channels=channel)
        mid = exp_ratio * channel
        self.mlp = Mlp(in_features=channel, hidden_features=mid, out_features=channel)

    def forward(self, x):
        x = self.norm(x)
        x = x + self.dila_conv(x)
        x = x + self.mlp(x)

        return x


# Rough Outline Generation Module (ROG)
class ROG_Module(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.down_x2 = Down(ch_in=channels[1], ch_out=channels[2])
        self.up_x4 = UP(ch_in=channels[3], ch_out=channels[2])
        self.dilation_conv = dilation_conv(channel=channels[2], exp_ratio=4)
        self.up_contour_x3 = UP(ch_in=channels[2], ch_out=channels[1])
        self.pwconv = ConvBNReLU(in_channel=3 * channels[2], out_channel=channels[2], kernel_size=1)

    def forward(self, x2, x3, x4):
        x2 = self.down_x2(x2)
        x4 = self.up_x4(x4)
        x_cat = self.pwconv(torch.cat((x3, (x3 * x2), (x3 * x4)), dim=1))
        x_cat = self.dilation_conv(x_cat)
        contour_x3 = x3 + x_cat
        contour_x2 = self.up_contour_x3(contour_x3)

        return contour_x2, contour_x3


# Edge Information Extraction Module (EIE)
class EIE_module(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.gate_fuse = GateFusion(in_planes=channel)
        self.local_feature = Local_conv(in_channel=channel, out_channel=channel,exp_ratio=4)

    def forward(self, x, edge_feature):
        xsize = x.size()[2:]
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        x_edge = x + x * edge_input

        x = self.gate_fuse(x, x_edge)
        x = self.local_feature(x)

        return x


# Outline Feature Clarifying Module (OFC)
class OFC_Module(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()

        self.pwconv = ConvBNReLU(in_channel=in_chs, out_channel=out_chs, kernel_size=1)
        self.se = SELayer(channel=out_chs, reduction=4)

    def forward(self, x):
        out = self.se(self.pwconv(x))

        return out


class LHONet_Model(nn.Module):
    def __init__(self, channels=[64, 128, 320, 512]):
        super().__init__()

        # Rough Outline Generation
        self.contour_feature = ROG_Module(channels)

        # Edge Information Extraction
        self.eie1 = EIE_module(channels[0])
        self.eie2 = EIE_module(channels[1])
        self.eie3 = EIE_module(channels[2])

        # Outline Feature Clarify
        self.fuse_x2 = OFC_Module(in_chs=2 * channels[1], out_chs=channels[1])
        self.fuse_x3 = OFC_Module(in_chs=3 * channels[2], out_chs=channels[2])

        self.clear1_down = Down(ch_in=channels[0], ch_out=channels[1])
        self.clear2_down = Down(ch_in=channels[1], ch_out=channels[2])

        self.up1 = UP(channels[1], channels[0])
        self.up2 = UP(channels[2], channels[1])
        self.up3 = UP(channels[3], channels[2])

        # Prediction heads initialization
        n_class = 1
        self.out_head1 = nn.Conv2d(channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(channels[3], n_class, 1)

    def forward(self, x1, x2, x3, x4, edge_feature):
        contour2, contour3 = self.contour_feature(x2, x3, x4)

        edge1 = self.eie1(x1, edge_feature)
        edge2 = self.eie2(x2, edge_feature)
        edge3 = self.eie3(x3, edge_feature)

        edge1_down_x2 = self.clear1_down(edge1)
        edge1_down_x4 = self.clear2_down(edge1_down_x2)
        edge2_down_x2 = self.clear2_down(edge2)
        clear_feature1 = torch.cat((contour2 + edge1_down_x2, contour2 + edge2), dim=1)
        clear_feature1 = self.fuse_x2(clear_feature1)
        clear_feature2 = torch.cat((contour3 + edge1_down_x4,  contour3 + edge2_down_x2, contour3 + edge3), dim=1)
        clear_feature2 = self.fuse_x3(clear_feature2)

        out2 = self.out_head2(clear_feature1)
        out3 = self.out_head3(clear_feature2)

        return out2, out3   # out1(1,64,88,88), out2(1,128,44,44), out3(1,320,22,22), out4(1,512,11,11)
