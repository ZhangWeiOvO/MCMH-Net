from CAFA import CAFA
from RDB import  RDB
from DCSMamba2 import DCSMamba2
import torch
import torch.nn.functional as F

from torch import LongTensor, Tensor, nn

class SobelOperator(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

        self.register_buffer('conv_x', torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None, None, :, :] / 4)
        self.register_buffer('conv_y', torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[None, None, :, :] / 4)

    def forward(self, x):
        b, c, h, w = x.shape
        if c > 1:
            x = x.view(b * c, 1, h, w)

        grad_x = F.conv2d(x, self.conv_x, bias=None, stride=1, padding=1)
        grad_y = F.conv2d(x, self.conv_y, bias=None, stride=1, padding=1)

        x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)

        x = x.view(b, c, h, w)

        return x



def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def bili_resize(factor):
    return nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)


##---------- Basic Blocks ----------
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.block = RDB(in_channels=in_size, growth_rate=out_size, num_layers=3)
        if downsample:
            self.downsample = PS_down(out_size, out_size, downscale=2)

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        # self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.up = PS_up(in_size, out_size, upscale=2)
        self.conv_block = UNetConvBlock(in_size, out_size, False)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], dim=1)
        out = self.conv_block(out)
        return out


##---------- Resizing Modules (Pixel(Un)Shuffle) ----------
class PS_down(nn.Module):
    def __init__(self, in_size, out_size, downscale):
        super(PS_down, self).__init__()
        self.UnPS = nn.PixelUnshuffle(downscale)
        self.conv1 = nn.Conv2d((downscale ** 2) * in_size, out_size, 1, 1, 0)

    def forward(self, x):
        x = self.UnPS(x)  # h/2, w/2, 4*c
        x = self.conv1(x)
        return x


class PS_up(nn.Module):
    def __init__(self, in_size, out_size, upscale):
        super(PS_up, self).__init__()

        self.PS = nn.PixelShuffle(upscale)
        self.conv1 = nn.Conv2d(in_size // (upscale ** 2), out_size, 1, 1, 0)

    def forward(self, x):
        x = self.PS(x)  # h/2, w/2, 4*c
        x = self.conv1(x)
        return x




##---------- MCMH_Net ----------
class MCMH_Net(nn.Module):
    def __init__(self, in_chn=1, wf=12, depth=3):
        super(MCMH_Net, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        self.bili_down = bili_resize(0.5)
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.connect_path = nn.ModuleList()
        self.conv_path = nn.ModuleList()
        # encoder of UNet
        prev_channels = wf
        for i in range(depth):  # 0,1,2,3
            downsample = True if (i + 1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels, (2 ** i) * wf, downsample))
            prev_channels = (2 ** i) * wf

        for i in range(depth):
            connect_channels = (2 ** i) * wf
            self.connect_path.append(CAFA(connect_channels))

        conv_channels = wf
        for i in range(depth):  # 0,1,2,3
            self.conv_path.append(nn.Conv2d(conv_channels, (2 ** i) * wf, 3, 1, 1))
            conv_channels = (2 ** i) * wf

        # decoder of UNet
        self.up_path = nn.ModuleList()
        self.skip_conv = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.bottom_conv = nn.Conv2d(prev_channels, wf, 3, 1, 1)
        self.bottom_up = bili_resize(2 ** (depth - 1))

        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2 ** i) * wf))
            self.skip_conv.append(nn.Conv2d((2 ** i) * wf, (2 ** i) * wf, 3, 1, 1))
            self.conv_up.append(nn.Sequential(*[nn.Conv2d((2 ** i) * wf, wf, 3, 1, 1), bili_resize(2 ** i)]))
            prev_channels = (2 ** i) * wf

        self.last = conv3x3(wf * depth, in_chn, bias=True)
        self.sobel = SobelOperator(epsilon=1e-4)
        self.last_manba = DCSMamba2(wf * depth)

    def forward(self, x):
        img = x
        scale_img = self.sobel(x)
        scale_img = self.conv_01(scale_img)
        ##### shallow conv #####
        x1 = self.conv_01(img)
        encs = []
        ######## UNet ########
        # Down-path (Encoder)
        for i, down in enumerate(self.down_path):
            if i == 0:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            elif (i + 1) < self.depth:
                scale_img = self.bili_down(scale_img)
                # x1 和 scale_img 维度相同
                scale_img = self.conv_path[i - 1](scale_img)
                x1 = self.connect_path[i - 1](x1, scale_img)
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:  # 最后一层
                scale_img = self.bili_down(scale_img)
                scale_img = self.conv_path[i - 1](scale_img)
                x1 = self.connect_path[i - 1](x1, scale_img)
                x1 = down(x1)

        # Up-path (Decoder)
        ms_result = [self.bottom_up(self.bottom_conv(x1))]
        for i, up in enumerate(self.up_path):
            x1 = up(x1, self.skip_conv[i](encs[-i - 1]))
            ms_result.append(self.conv_up[i](x1))
        x1 = torch.cat(ms_result, dim=1)
        x1 = self.last_manba(x1)

        out_1 = self.last(x1) + img

        return out_1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.ones(1, 1, 64, 64, dtype=torch.float, requires_grad=False).to(device)

    model = MCMH_Net().to(device)
    out = model(input)
    print(out.shape)


