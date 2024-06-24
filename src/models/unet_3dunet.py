"""A small Unet-like zoo"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from src.models.layers import ConvBnRelu, UBlock, conv1x1, UBlockCbam, CBAM, MultiConv, MultiConvRes


class Unet(nn.Module):
    """Almost the most basic U-net.
    """
    name = "Unet"

    def __init__(self, inplanes, num_classes, width=32, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()

        # 参数配置
        self.in_channels = inplanes
        self.n_classes = num_classes

        # self.num = [width * 2 ** i for i in range(5)]  # [32, 64, 128, 256, 512]
        self.num = width

        self.relu = nn.ReLU()
        # self.dropout3d = nn.Dropout3d(p=0.6)
        # self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()
        self.deep_supervision = deep_supervision

        # 左侧卷积
        self.L1_conv1 = nn.Conv3d(self.in_channels, self.num * 1, kernel_size=3, stride=1, padding=1, bias=False)  # 4-32
        self.L1_bn1 = nn.BatchNorm3d(self.num * 1)
        self.L1_conv2 = nn.Conv3d(self.num * 1, self.num * 2, kernel_size=3, stride=1, padding=1, bias=False)  # 32-64
        self.L1_bn2 = nn.BatchNorm3d(self.num * 2)

        self.L2_conv1 = nn.Conv3d(self.num * 2, self.num * 2, kernel_size=3, stride=1, padding=1, bias=False)  # 64-64
        self.L2_bn1 = nn.BatchNorm3d(self.num * 2)
        self.L2_conv2 = nn.Conv3d(self.num * 2, self.num * 4, kernel_size=3, stride=1, padding=1, bias=False)  # 64-128
        self.L2_bn2 = nn.BatchNorm3d(self.num * 4)

        self.L3_conv1 = nn.Conv3d(self.num * 4, self.num * 4, kernel_size=3, stride=1, padding=1, bias=False)  # 128-128
        self.L3_bn1 = nn.BatchNorm3d(self.num * 4)
        self.L3_conv2 = nn.Conv3d(self.num * 4, self.num * 8, kernel_size=3, stride=1, padding=1, bias=False)  # 128-256
        self.L3_bn2 = nn.BatchNorm3d(self.num * 8)

        self.L4_conv1 = nn.Conv3d(self.num * 8, self.num * 8, kernel_size=3, stride=1, padding=1, bias=False)  # 256-256
        self.L4_bn1 = nn.BatchNorm3d(self.num * 8)
        self.L4_conv2 = nn.Conv3d(self.num * 8, self.num * 16, kernel_size=3, stride=1, padding=1, bias=False)  # 256-512
        self.L4_bn2 = nn.BatchNorm3d(self.num * 16)

        # 左侧下采样
        self.L12_down1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 边长减半
        self.L23_down1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L34_down1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # 右侧卷积
        self.R3_conv1 = nn.Conv3d(self.num * (8 + 16), self.num * 8, kernel_size=3, stride=1, padding=1, bias=False)  # 768-256
        self.R3_bn1 = nn.BatchNorm3d(self.num * 8)
        self.R3_conv2 = nn.Conv3d(self.num * 8, self.num * 8, kernel_size=3, stride=1, padding=1, bias=False)  # 256-256
        self.R3_bn2 = nn.BatchNorm3d(self.num * 8)

        self.R2_conv1 = nn.Conv3d(self.num * (4 + 8), self.num * 4, kernel_size=3, stride=1, padding=1, bias=False)  # 384-128
        self.R2_bn1 = nn.BatchNorm3d(self.num * 4)
        self.R2_conv2 = nn.Conv3d(self.num * 4, self.num * 4, kernel_size=3, stride=1, padding=1, bias=False)  # 128-128
        self.R2_bn2 = nn.BatchNorm3d(self.num * 4)

        self.R1_conv1 = nn.Conv3d(self.num * (2 + 4), self.num * 2, kernel_size=3, stride=1, padding=1, bias=False)  # 192-64
        self.R1_bn1 = nn.BatchNorm3d(self.num * 2)
        self.R1_conv2 = nn.Conv3d(self.num * 2, self.num * 2, kernel_size=3, stride=1, padding=1, bias=False)  # 64-64
        self.R1_bn2 = nn.BatchNorm3d(self.num * 2)
        self.R1_conv3 = nn.Conv3d(self.num * 2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)  # 64-5

        # 右侧上采样
        self.R43_up1 = nn.ConvTranspose3d(self.num * 16, self.num * 16, kernel_size=2, stride=2)
        self.R32_up1 = nn.ConvTranspose3d(self.num * 8, self.num * 8, kernel_size=2, stride=2)
        self.R21_up1 = nn.ConvTranspose3d(self.num * 4, self.num * 4, kernel_size=2, stride=2)

    def forward(self, x):
        # 左侧第一层
        x1 = self.L1_conv1(x)  # 第一次卷积
        x1 = self.L1_bn1(x1)
        x1 = self.relu(x1)
        x1 = self.L1_conv2(x1)  # 第二次卷积
        x1 = self.L1_bn2(x1)
        x1 = self.relu(x1)
        x1_out = x1  # 待拼接

        x1 = self.L12_down1(x1)  # 下采样进入第二层

        # 左侧第二层
        x2 = self.L2_conv1(x1)  # 第一次卷积
        x2 = self.L2_bn1(x2)
        x2 = self.relu(x2)
        x2 = self.L2_conv2(x2)  # 第二次卷积
        x2 = self.L2_bn2(x2)
        x2 = self.relu(x2)
        x2_out = x2  # 待拼接

        x2 = self.L23_down1(x2)  # 下采样进入第三层

        # 左侧第三层
        x3 = self.L3_conv1(x2)  # 第一次卷积
        x3 = self.L3_bn1(x3)
        x3 = self.relu(x3)
        x3 = self.L3_conv2(x3)  # 第二次卷积
        x3 = self.L3_bn2(x3)
        x3 = self.relu(x3)
        x3_out = x3  # 待拼接

        x3 = self.L34_down1(x3)  # 下采样进入第四层

        # 左（右）侧第四层
        x4 = self.L4_conv1(x3)  # 第一次卷积
        x4 = self.L4_bn1(x4)
        x4 = self.relu(x4)
        x4 = self.L4_conv2(x4)  # 第二次卷积
        x4 = self.L4_bn2(x4)
        x4 = self.relu(x4)

        y3 = self.R43_up1(x4)  # 上采样进入第三层
        y3_in = y3  # 待拼接

        # 右侧第三层
        y3 = torch.cat([x3_out, y3_in], dim=1)
        y3 = self.R3_conv1(y3)  # 第一次卷积
        y3 = self.R3_bn1(y3)
        y3 = self.relu(y3)
        y3 = self.R3_conv2(y3)  # 第二次卷积
        y3 = self.R3_bn2(y3)
        y3 = self.relu(y3)

        y3 = self.R32_up1(y3)  # 上采样进入第二层
        y2_in = y3  # 待拼接

        # 右侧第二层
        y2 = torch.cat([x2_out, y2_in], dim=1)
        y2 = self.R2_conv1(y2)  # 第一次卷积
        y2 = self.R2_bn1(y2)
        y2 = self.relu(y2)
        y2 = self.R2_conv2(y2)  # 第二次卷积
        y2 = self.R2_bn2(y2)
        y2 = self.relu(y2)

        y2 = self.R21_up1(y2)  # 上采样进入第一层
        y1_in = y2  # 待拼接

        # 右侧第一层
        y1 = torch.cat([x1_out, y1_in], dim=1)
        y1 = self.R1_conv1(y1)  # 第一次卷积
        y1 = self.R1_bn1(y1)
        y1 = self.relu(y1)
        y1 = self.R1_conv2(y1)  # 第二次卷积
        y1 = self.R1_bn2(y1)
        y1 = self.relu(y1)
        y1 = self.R1_conv3(y1)  # 第三次卷积

        # seg_layer = self.sigmoid(y1)
        # seg_layer = self.softmax(y1)

        seg_layer = y1
        return seg_layer


class EquiUnet(Unet):
    """Almost the most basic U-net: all Block have the same size if they are at the same level.
    """
    name = "EquiUnet"
    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)


class Att_EquiUnet(Unet):
    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False,  dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)


# 测试网络是否通
if __name__ == "__main__":
    # 计时
    import time
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 128
    x = torch.Tensor(1, 4, image_size, image_size, image_size)
    x.to(device)
    print("x size: {}".format(x.size()))
    model = Unet(inplanes=4, num_classes=3, width=32)
    out = model(x)
    print("out size: {}".format(out.size()))

    # 计时
    end_time = time.time()
    cost_time = end_time - start_time
    print(f"代码执行时间：{cost_time:.2f}秒")
