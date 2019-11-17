import torch
from torch import nn


class DropoutFreeze(nn.Dropout):
    def __init__(self, p=0.5, inplace=False, name='drop'):
        super(DropoutFreeze, self).__init__(p, inplace)
        self.name = name
        self.freeze = False
        self.p = p

    def forward(self, x, mask=None):
        if self.training and self.freeze:
            # use mask from outside
            assert mask.shape == x.shape
            x = x.mul(mask)
        else:
            # in eval model, or in training mode but not freeze
            if self.training:
                # training mode
                self.mask = torch.zeros_like(x).bernoulli_(1. - self.p).div_(1. - self.p)
                x = x.mul(self.mask)
            else:
                # eval mode
                pass
        return x
