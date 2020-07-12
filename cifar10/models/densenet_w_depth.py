import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = ['densenet']

"""
densenet with basic block.
"""
class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    """
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
        """
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,))
        output = input_tensor[:, selected_index, :, :]
        return output

class BasicBlock(nn.Module):
    def __init__(self, inplanes, cfg, expansion=1, growthRate=12, dropRate=0): # an object in Python if the function I am doing this from is the base class of which the class of the instance has been derived?
# Was thinking maybe the inspect module might have helped me out here, but it doesn't seem to give me what I want. And short of parsing the __class__ member, I'm not sure how to get at this information.

        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes, cfg):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

class densenet(nn.Module):

    def __init__(self, depth=40,
        dropRate=0, dataset='cifar10', growthRate=12, compressionRate=1, cfg = None):
        super(densenet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3
        block = BasicBlock

        self.growthRate = growthRate
        self.dropRate = dropRate

        if cfg == None:
            cfg = []
            start = growthRate*2
            for _ in range(3):
                cfg.append([start + growthRate*i for i in range(n+1)])
                start += growthRate*n
            cfg = [item for sub_list in cfg for item in sub_list]

        assert len(cfg) == 3*n+3, 'length of config variable cfg should be 3n+3'

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_denseblock(block, n, cfg[0:n])
        self.trans1 = self._make_transition(compressionRate, cfg[n])
        self.dense2 = self._make_denseblock(block, n, cfg[n+1:2*n+1])
        self.trans2 = self._make_transition(compressionRate, cfg[2*n+1])
        self.dense3 = self._make_denseblock(block, n, cfg[2*n+2:3*n+2])
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.select = channel_selection(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg[-1], 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg[-1], 100)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, cfg):
        layers = []
        assert blocks == len(cfg), 'Length of the cfg parameter is not right.'
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, cfg = cfg[i], growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate, cfg):
        # cfg is a number in this case.
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes, cfg)

    def forward(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def DenseNet40():
    return densenet()