import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import DropoutFreeze

__all__ = ['mlp', 'mlp800']


class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 150)
        self.fc3 = nn.Linear(150, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        self.x = x
        self.flat_out = self.x.view(-1, 784)
        self.fc1_out = self.fc1(self.flat_out)
        self.relu1_out = F.relu(self.fc1_out)
        self.fc2_out = self.fc2(self.relu1_out)
        self.relu2_out = F.relu(self.fc2_out)
        self.fc3_out = self.fc3(self.relu2_out)
        return self.fc3_out


class MLP800(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP800, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, num_classes)
        self.drop1 = DropoutFreeze(name='drop1')
        self.drop2 = DropoutFreeze(name='drop2')

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x, **masks):
        self.x = x
        self.flat_out = self.x.view(-1, 784)
        self.fc1_out = self.drop1(self.fc1(self.flat_out), mask=masks.get('drop1_mask', None))
        self.relu1_out = F.relu(self.fc1_out)
        self.fc2_out = self.drop2(self.fc2(self.relu1_out), mask=masks.get('drop2_mask', None))
        self.relu2_out = F.relu(self.fc2_out)
        self.fc3_out = self.fc3(self.relu2_out)
        return self.fc3_out


def mlp(**kwargs):
    return MLP(**kwargs)


def mlp800(**kwargs):
    return MLP800(**kwargs)

