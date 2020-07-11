import torch
import torch.nn as nn
import numpy as np
from models import *

class SizeEstimator(object):

    def __init__(self, model, input_size=(1, 3, 224, 224)):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.model = model.cpu()
        self.input_size = input_size
        return

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `model`'''
        sizes=sum([p.numel() for p in self.model.parameters()])

        return sizes

    def check_modules(self, module_name):
        modules_in_eff = ['conv', 'bn', 'se', 'swish', 'fc', 'pool', 'dropout']
        for module in modules_in_eff:
            if module in module_name:
                return True
        return False

    def check_modules_for_act(self, module_name):
        modules_in_eff = ['conv', 'bn', 'se', 'fc', 'pool']

        modules_to_pass = ['padding', 'drop', 'swish']
        for module in modules_to_pass:
            if module in module_name:
                return False

        for module in modules_in_eff:
            if module in module_name:
                return True
        return False

    def get_output_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        input_ = torch.FloatTensor(*self.input_size)

        mods = list()

        for n, m in self.model.named_modules():
            if not self.check_modules_for_act(n):
                continue
            mods.append(m)
        out_sizes = []

        for i in range(len(mods)):
            m = mods[i]
            if isinstance(m, nn.Linear):
                input_ = input_.view(self.input_size[0], -1)
            # print(colored(input_.shape, 'green'))
            out = m(input_)
            out_sizes.append(np.array(out.detach().cpu()))
            print(out.shape)
            input_ = out

        total_activation = 0

        self.out_sizes = out_sizes
        for i in range(len(out_sizes)):
            s = out_sizes[i]
            act = np.count_nonzero(s)

            total_activation += act

        return total_activation


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = EfficientNetB0()
    # net = net.to(device)
    # net = torch.nn.DataParallel(net)
    # checkpoint = torch.load('ResNet95.57.pth')
    # net.load_state_dict(checkpoint['net'])
    se = SizeEstimator(net, input_size=(1,3,32,32))
    act = se.get_output_sizes()
    param = se.get_parameter_sizes()
    print(act+param)