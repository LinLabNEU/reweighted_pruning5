'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=100):
        super(MobileNetV2, self).__init__()
        # (expansion, out_planes, num_blocks, stride)
        self.cfg = [(1, self.expanding(v=16), 1, 1),
               (6, self.expanding(v=24), 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
               (6, self.expanding(v=32), 3, 2),
               (6, self.expanding(v=64), 4, 2),
               (6, self.expanding(v=96), 3, 1),
               (6, self.expanding(v=160), 3, 2),
               (6, self.expanding(v=320), 1, 1)]

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, self.expanding(v=32), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.expanding(v=32))
        self.layers = self._make_layers(in_planes=self.expanding(v=32))
        last_conv_output = max(1280, self.expanding(v=1280))
        self.conv2 = nn.Conv2d(self.expanding(v=320), last_conv_output, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(last_conv_output)
        self.linear = nn.Linear(last_conv_output, num_classes)

    @staticmethod
    def expanding(v, divisor=8, min_value=None):  # add function of expanding network
        model_exp = 1.3  # expanding paramenter
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v * model_exp + divisor / 2) // divisor * divisor)
        return new_v

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
    print(net)


# test()


