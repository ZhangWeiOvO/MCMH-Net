import torch
import torch.nn as nn
import torch.nn.functional as F

##---------- Dense Block ----------
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, I):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.manba = DCSMamba2(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.relu(self.conv(x))
        output = self.manba(x1)
        return output + x


##---------- Residual DCSMamba2 Block (RDB) ----------
class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.identity = nn.Conv2d(in_channels, growth_rate, 1, 1, 0)
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels, in_channels, I=i) for i in range(num_layers)]
        )
        # self.manba = DCSMamba2(in_channels, 128)
        self.lff = nn.Conv2d(in_channels, growth_rate, kernel_size=1)

    def forward(self, x):
        res = self.identity(x)
        x = self.layers(x)
        x = self.lff(x)
        return res + x

