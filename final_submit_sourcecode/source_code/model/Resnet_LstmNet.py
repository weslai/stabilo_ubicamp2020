import torch
import torch.nn as nn
import math


def conv1d(in_planes, out_planes, kernel_size= 5, stride=1):
    """1d convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size= kernel_size, stride= stride,
                     padding=kernel_size//2, bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample= None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d(planes, planes, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.conv1x1 = conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        residual = self.conv1x1(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottlenneck(nn.Module): ## 3 Conv layers
    expansion = 4

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(Bottlenneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size//2, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_lstm(nn.Module):
    """ResNet 50"""
    def __init__(self, block, layers, num_classes=52):
        self.inplanes = 32
        super(ResNet_lstm, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=32, kernel_size=7, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(block, 32, layers[0], kernel_size=7, stride=1)
        self.layer1 = self._make_layer(block, 32, layers[0], kernel_size=7, stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], kernel_size=7, stride=1)
        self.layer3 = self._make_layer(block, 64, layers[1], kernel_size=7, stride=1)
        self.layer4 = self._make_layer(block, 64, layers[2], kernel_size=5, stride=1)
        self.layer5 = self._make_layer(block, 64, layers[2], kernel_size=5, stride=1)
        self.layer6 = self._make_layer(block, 128, layers[3], kernel_size=3, stride=1)
        self.layer7 = self._make_layer(block, 128, layers[3], kernel_size=3, stride=1)

        # self.conv_merge = nn.Conv1d(512 * block.expansion, num_classes, kernel_size=3, stride=1,
        #                             padding=1, bias=True)
        # self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.rnn = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, dropout=0.2,
                           batch_first=True)  ## input_size = expected features
        # self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(256, num_classes)

        ## initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1):
        """
        Args:
            block: the block to use (Bottleneck, basicblock)
            planes: the number of kernels, not the kernel_size
            blocks: the number to repeat for the block
            kernel_size: kernel size of the convolution layer
            stride: stride of the convolution layer

        Returns:

        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=kernel_size, stride=1, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size= kernel_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        x = torch.transpose(x, 1, 2)  ## x return [Batch, feature_size, kernel_sizes]
        x, (h, c) = self.rnn(x)  ## h_n return [Batch, num_layers, hidden_size]
        # x = self.dropout(_)
        x = torch.transpose(h, 0, 1)  ## x return [Batch, num_layers, num_classes]
        x = self.dropout(x)
        x = self.linear(x)
        x = x[:, 0, :]  ## x return [Batch, num_classes]
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x