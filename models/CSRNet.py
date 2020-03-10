import torch.nn as nn
import torch
from torchvision import models

"""
    2 Classes to create a complete CSRNet Model
"""


class CSRNet(nn.Module):
    """Front Layers of a CSNet model

        Args:
        pre_trained (bool): If True, load layers from VGG
    """

    def __init__(self, backend, pre_trained=True):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128,
                              'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = backend
        self.seen = 0

        # If pre-trained load weights from VGG
        if pre_trained:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[
                    i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CSRNet_Back(nn.Module):
    """Backend Layers of a CSNet model

        Args:
        dilation (List): If given, the dilation rate applied for each layer of the backend, else 1
    """

    def __init__(self, dilation=None):
        super(CSRNet_Back, self).__init__()

        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.dilation = dilation

        self.backend = make_layers(
            self.backend_feat, in_channels=512, dilation=self.dilation)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.backend(x)
        x = self.output_layer(x)
        return x


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=None):
    if dilation == None:
        dilation = [1]*len(cfg)
    layers = []
    for v, d_rate in zip(cfg, dilation):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d,
                           nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
