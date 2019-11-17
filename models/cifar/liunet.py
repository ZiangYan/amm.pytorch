import torch.nn as nn

__all__ = ['liunet']


class LiuNet(nn.Module):
    def __init__(self, width=64, num_classes=10, num_group=16):
        super(LiuNet, self).__init__()

        self.num_group = num_group

        if width == 64:
            out_channels = [64, 64, 96, 128, 256]
        elif width == 96:
            out_channels = [96, 96, 192, 384, 512]
        else:
            raise NotImplementedError

        # conv0.x
        self.layer0 = self.make_layer(num_layers=1, in_channels=3, out_channels=out_channels[0])

        # conv1.x
        self.layer1 = self.make_layer(num_layers=4, in_channels=out_channels[0], out_channels=out_channels[1])

        # conv2.x
        self.layer2 = self.make_layer(num_layers=4, in_channels=out_channels[1], out_channels=out_channels[2])

        # conv3.x
        self.layer3 = self.make_layer(num_layers=4, in_channels=out_channels[2], out_channels=out_channels[3])

        # fc
        self.layer4 = nn.Sequential(nn.Linear(out_channels[3] * 4 * 4, out_channels[4]),
                                    nn.ReLU(),
                                    nn.Linear(out_channels[4], num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, num_layers=3, in_channels=64, out_channels=64):
        layers = list()
        for i in range(num_layers):
            if i > 0:
                in_channels = out_channels
            layers += [nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                       # nn.BatchNorm2d(out_channels),
                       nn.GroupNorm(self.num_group, out_channels),
                       nn.ReLU()]
        if num_layers > 1:
            layers.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*layers)

    def forward(self, x, freeze=False):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        return x


def liunet(width, **kwargs):
    return LiuNet(width, **kwargs)

