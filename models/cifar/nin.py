import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import DropoutFreeze

__all__ = ['nin']


class NIN(nn.Module):
    def __init__(self, num_classes=10):
        super(NIN, self).__init__()
        self.conv1 = nn.Conv2d(3, 192, 5, padding=2)
        self.cccp1 = nn.Conv2d(192, 160, 1)
        self.cccp2 = nn.Conv2d(160, 96, 1)
        self.drop1 = DropoutFreeze(name='drop1')

        self.conv2 = nn.Conv2d(96, 192, 5, padding=2)
        self.cccp3 = nn.Conv2d(192, 192, 1)
        self.cccp4 = nn.Conv2d(192, 192, 1)
        self.drop2 = DropoutFreeze(name='drop2')

        self.conv3 = nn.Conv2d(192, 192, 3, padding=1)
        self.cccp5 = nn.Conv2d(192, 192, 1)
        self.cccp6 = nn.Conv2d(192, num_classes, 1)

        for k in ['conv1', 'cccp1', 'cccp2', 'conv2', 'cccp3', 'cccp4', 'conv3', 'cccp5', 'cccp6']:
            w = self.__getattr__(k)
            torch.nn.init.kaiming_normal_(w.weight.data)
            w.bias.data.fill_(0)

        # self.cccp6.weight.data[:] = 0.1 * self.cccp6.weight.data[:]

        self.p = 0.5  # dropout probability

    def forward(self, x, **masks):
        self.x = x
        self.conv1_out = self.conv1(self.x)
        self.relu1_out = F.relu(self.conv1_out)
        self.cccp1_out = self.cccp1(self.relu1_out)
        self.relu_cccp1_out = F.relu(self.cccp1_out)
        self.cccp2_out = self.cccp2(self.relu_cccp1_out)
        self.relu_cccp2_out = F.relu(self.cccp2_out)
        self.pool1_out, self.pool1_ind = F.max_pool2d(self.relu_cccp2_out, kernel_size=2, stride=2, return_indices=True)
        self.drop1_out = self.drop1(self.pool1_out, mask=masks.get('drop1_mask', None))

        self.conv2_out = self.conv2(self.drop1_out)
        self.relu2_out = F.relu(self.conv2_out)
        self.cccp3_out = self.cccp3(self.relu2_out)
        self.relu_cccp3_out = F.relu(self.cccp3_out)
        self.cccp4_out = self.cccp4(self.relu_cccp3_out)
        self.relu_cccp4_out = F.relu(self.cccp4_out)
        self.pool2_out = F.avg_pool2d(self.relu_cccp4_out, kernel_size=2, stride=2)
        self.drop2_out = self.drop2(self.pool2_out, mask=masks.get('drop2_mask', None))

        self.conv3_out = self.conv3(self.drop2_out)
        self.relu3_out = F.relu(self.conv3_out)
        self.cccp5_out = self.cccp5(self.relu3_out)
        self.relu_cccp5_out = F.relu(self.cccp5_out)
        self.cccp6_out = self.cccp6(self.relu_cccp5_out)
        self.pool3_out = F.avg_pool2d(self.cccp6_out, kernel_size=8)
        self.flat_out = self.pool3_out.view(self.x.shape[0], -1)
        return self.flat_out


def nin(**kwargs):
    return NIN(**kwargs)
