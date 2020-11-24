import torch
import torch.nn as nn
import math

### conv1d
def conv1d(in_planes, out_planes, kernel_size=5, stride=1, padding=1, padding_mode='zeros'):
    """1d convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, padding_mode=padding_mode)

class Cnn_RnnNet(nn.Module):
    def __init__(self, inplanes, outplanes, num_classes=52):
        super(Cnn_RnnNet, self).__init__()
        self.conv1 = conv1d(inplanes, outplanes, kernel_size=11, stride=2)
        self.conv2 = conv1d(outplanes, outplanes, kernel_size=11, stride=1)
        self.conv3 = conv1d(outplanes, outplanes, kernel_size=11, stride=1)
        # self.conv4 = conv1d(2 * outplanes, 2 * outplanes, kernel_size=7, stride=1)
        # self.exponential_activation = torch.exp()
        self.conv5 = conv1d(outplanes,  2 * outplanes, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(2 * outplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool1 = nn.MaxPool1d(kernel_size=7, stride=3)
        self.conv6 = conv1d(2 * outplanes, 2 * outplanes, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(2 * outplanes)
        self.maxpool2 = nn.MaxPool1d(kernel_size=7, stride=3)
        self.conv7 = conv1d(2 * outplanes, 4 * outplanes, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(4 * outplanes)
        self.conv8 = conv1d(4 * outplanes, 4 * outplanes, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm1d(4 * outplanes)
        self.conv9 = conv1d(4 * outplanes, 4 * outplanes, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm1d(4 * outplanes)
        # self.avg_pool_global = nn.AdaptiveAvgPool1d(1)
        self.rnn = nn.GRU(input_size= 4 * outplanes, hidden_size= 8 * outplanes, num_layers= 1, dropout=0.2,
                          batch_first=True) ## input_size = expected features
        # self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(8 * outplanes, num_classes)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)
        # x = self.relu(x)
        x = self.conv2(x)
        # x = self.relu(x)
        x = self.conv3(x)
        # x = self.relu(x)
        # x = self.conv4(x)
        x = torch.exp(x)
        x = self.conv5(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv6(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.conv7(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.bn5(x)
        x = self.relu(x)
        # x = self.avg_pool_global(x)
        # x = torch.flatten(x, 1)
        x = torch.transpose(x, 1, 2) ## x return [Batch, feature_size, kernel_sizes]
        x, _ = self.rnn(x) ## h_n return [Batch, num_layers, hidden_size]
        # x = self.dropout(_)
        x = torch.transpose(_, 0, 1) ## x return [Batch, num_layers, num_classes]
        x = self.fc(x)
        x = x[:, 0, :] ## x return [Batch, num_classes]
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x ## (batch_size, num_classes)