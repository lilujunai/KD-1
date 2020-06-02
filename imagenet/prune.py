from torch.nn.utils.prune import BasePruningMethod
import torch.nn.utils.prune as prune

class L1Pruning(BasePruningMethod):
    def __init__(self, ratio):
        self.ratio = ratio
    def compute_mask(self, t, default_mask):
        t_flat = t.reshape(3,-1)
        t_shape = t_flat.shape[1]
        t_sorted = t_flat.sort(dim=1, descending=False)
        threshold = t_sorted[int(t_shape * self.ratio)]
        for batch, element in t_flat:

def prune_l1(model, amount=0.2):
    block_len = len(model.module._blocks)
    parameters_to_prune = ()
    for i in range(12):
        parameters_to_prune += (
            (model.module.layer1[0].conv1, 'weight')
        )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.5,
    )
    model = prune.l1_unstructured(model, name='weight', amount=amount)

    return model


import time
from tqdm import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

class pruning(nn.Module):
    def __init__(self, r_delete = 0.3, r_fisher = 0.9, _fisher_ = True, _mag_ = True, lam = 0.01, thres = 20, boostType = 'naive', device = 'cuda'):
        """
        c = 64
        filter size : 64*3*3 => 576
        r_delete : 0.5 => 288
        r_fisher = (28.8 param per 0.1 point)
        """
        super(pruning, self).__init__()
        self.ratio_of_remove = r_delete
        if _fisher_ is True and _mag_ is True:
            self.ratio_of_fisher = r_fisher
        elif _fisher_ is True:
            self.ratio_of_fisher = 1
        elif _mag_ is True:
            self.ratio_of_fisher = 0
        else :
            self.ratio_of_fisher = -1
        self.device = device
        self.fisher__ = _fisher_
        self.mag__ = _mag_
        self.pruned_dict = None
        self.KL_matrix = None
        self.lam = lam
        self.thres = 10
        self.boostType = boostType
    def print_param(self):
        print('------------------------------')
        print('pruning parameters!')
        print('------------------------------')
        print('ratio_of_remove      %s' % self.ratio_of_remove)
        print('fisher               %s' % self.ratio_of_fisher)
        print('is_pruned_dict       %s' % (self.pruned_dict != None))
        print('is_KL_matrix         %s' % (self.KL_matrix != None))
        print('boostType            %s' % self.boostType)
        print('boostEpoch           %s' % self.thres)
        print('lambda               %s' % self.lam)
        print('device               %s' % self.device)
        print('-'*30)
    def cal_KLloss_p(self, orgnet, prunnet, DataLoader):
        orgnet.apply(self.filter_bn_down)
        prunnet.apply(self.filter_bn_down)
        KL_matrix = {}
        KLloss = torch.nn.KLDivLoss(size_average = False)
        for n1,p1 in prunnet.named_parameters():
            if p1.requires_grad:
                KL_matrix[n1] = 0
        #print(prunnet)
        for data in DataLoader:
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            prunnet.zero_grad()
            target = orgnet(images)
            output = prunnet(images)[:, :target.size(1)]
            loss = KLloss(F.log_softmax(output, dim=1), F.softmax(target, dim=1))
            loss.backward()
            for n2, p2 in prunnet.named_parameters():
                if p2.requires_grad:
                    KL_matrix[n2] += p2.grad.data
        prunnet.zero_grad()
        self.KL_matrix = KL_matrix
        orgnet.apply(self.filter_bn_on)
        prunnet.apply(self.filter_bn_on)
        return KL_matrix
    def boost_grad_with_KL(self, net):
        if self.pruned_dict is None:
            print("error, NO pruned_dict")
        if self.KL_matrix is None:
            print("error, NO KL_matrix")
        #sum_bf = 0
        #sum_af = 0
        for n2,p2 in net.named_parameters():
            if n2 in self.pruned_dict.keys():
                sum_bf += p2.grad.data[self.pruned_dict[n2]].abs().mean()
                sign = ((p2.grad.data[self.pruned_dict[n2]] > 0).float() - 0.5) * 2
                p2.grad.data[self.pruned_dict[n2]] += self.KL_matrix[n2][self.pruned_dict[n2]] * sign
                sum_af += p2.grad.data[self.pruned_dict[n2]].abs().mean()
        #print('bf', sum_bf/ len(self.pruned_dict.keys()))
        #print('af', sum_af/ len(self.pruned_dict.keys()))
    def boost_grad_mul(self, net):
        if self.pruned_dict is None:
            print("error, NO pruned_dict")
        for n2,p2 in net.named_parameters():
            if n2 in self.pruned_dict.keys():
                p2.grad.data[self.pruned_dict[n2]] *= 10
    #call after loss.backward() before optimizer.step()
    def boost_grad_naive(self, net):
        if self.pruned_dict is None:
            print("error, NO pruned_dict")
        for n2,p2 in net.named_parameters():
            if n2 in self.pruned_dict.keys():
                sign = ((p2.grad.data[self.pruned_dict[n2]] > 0).float() - 0.5) * 2
                p2.grad.data[self.pruned_dict[n2]] += self.lam * sign
    def boost_grad(self, net, epoch):
        if epoch < self.thres:
            if self.boostType == 'naive':
                #print('naive mode in')
                self.boost_grad_naive(net)
                #print('naive mode out')
            elif self.boostType == 'mul':
                #print('mul mode in')
                self.boost_grad_mul(net)
                #print('mul mode out')
            elif self.boostType == 'KL':
                #print('KL mode in')
                self.boost_grad_with_KL(net)
                #print('KL mode out')
            elif self.boostType == 'none':
                print("none")
            else:
                print("boosting mode error")
    def cal_fisher(self, net, DataLoader):
        net.zero_grad()
        fisher_matrix = {}
        net.eval()
        for n1,p1 in net.named_parameters():
            if p1.requires_grad:
                fisher_matrix[n1] = 0
        for data in DataLoader:
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            net.zero_grad()
            output = net(images)
            loss = F.cross_entropy(output, labels.long())
            loss.backward()
            for n2,p2 in net.named_parameters():
                if p2.requires_grad:
                    fisher_matrix[n2] += p2.grad.data ** 2 / len(DataLoader.dataset)
        return fisher_matrix
    def filter_bn_down(self, m):
        if isinstance(m, nn.BatchNorm2d):
            for param in m.parameters():
                param.requires_grad = False
    def filter_bn_on(self, m):
        if isinstance(m, nn.BatchNorm2d):
            for param in m.parameters():
                param.requires_grad = True
    def find_old_params(self, net, old_fisher):
        net.apply(self.filter_bn_down)
        old_params = {}
        old_params_linear = {}
        old_fisher_linear = {}
        for name, param in net.named_parameters():
            if param.requires_grad != False:
                old_params[name] = torch.tensor(param)
            else:
                try:
                    del old_fisher[name]
                except:
                    continue
        net.apply(self.filter_bn_on)
        return old_params
    def pruned_weights(self, old_params, old_fisher, task_num, mode = 'channel', see_cnt = True):
        cnt_a = 0
        cnt_f = 0
        cnt_m = 0
        cnt_cnn = 0
        for key in old_params.keys():
            cnt_a += len(old_params[key].view(-1))
            if len(old_params[key].size()) == 4:
                out_c, in_c, h, w =  old_params[key].shape
                if mode == 'channel':
                    len_weight = in_c * h * w
                    params_view = old_params[key].view((out_c, in_c * h * w))
                    fisher_view = old_fisher[key].view((out_c, in_c * h * w))
                #Currently not working
                elif mode == 'element':
                    len_weight = h * w
                    params_view = old_params[key].view((out_c * in_c, h * w))
                    fisher_view = old_fisher[key].view((out_c * in_c, h * w))
                cnt_cnn += len(old_params[key].view(-1))
            elif len(old_params[key].size()) == 2:
                #print(key) linear weights
                len_weight = old_params[key].size(1)
                params_view = old_params[key][:task_num*10]
                fisher_view = old_fisher[key][:task_num*10]
            elif len(old_params[key].size()) == 1:
                len_weight = old_params[key][:task_num*10].size(0)
                params_view = old_params[key][:task_num*10].view(1,-1)
                fisher_view = old_fisher[key][:task_num*10].view(1,-1)
            else :
                print('dim error! dim must be 1 or 4')
            ##################################################################
            #how many param will be deleted
            len_remove = int(len_weight*self.ratio_of_remove)
            #how many param will be deleted from fisher
            fish_remove = int(len_remove*self.ratio_of_fisher) if self.ratio_of_fisher != -1 else 0
            #how many param will be deleted from mag
            mag_remove = int(len_remove-fish_remove) if self.ratio_of_fisher != -1 else 0
            ###############################################################
            if self.mag__ == True:
                max_of_fisherV = fisher_view.max() #when try fisher prune, ignore index already set to 0.
                #_, mag_order = params_view.abs().sort(dim=1, descending = True)
                _, mag_order = params_view.abs().sort(dim=1, descending = False)
                for iter_num, iter_order in enumerate(mag_order):
                    params_view[iter_num][iter_order[:mag_remove]] = 0
                    fisher_view[iter_num][iter_order[:mag_remove]] = max_of_fisherV
                    cnt_m += len(params_view[iter_num][iter_order[:mag_remove]])
            if self.fisher__ == True:
                _, fish_order = fisher_view.sort(dim=1, descending = False)
                for iter_num, iter_order in enumerate(fish_order):
                    params_view[iter_num][iter_order[:fish_remove]] = 0
                    cnt_f += len(params_view[iter_num][iter_order[:fish_remove]])
        if see_cnt == True:
            print('-'*16,'pruning', '-'*16)
            print('number of all parameters : %10d(%5f %%)'%(cnt_a, 100))
            print('number of deleted params : %10d(%5f %%)'%(cnt_f+cnt_m,(cnt_f+cnt_m)/cnt_a*100))
            print('number of deleted magnit : %10d(%5f %%)'%(cnt_m, cnt_m/cnt_a*100))
            print('number of deleted fisher : %10d(%5f %%)'%(cnt_f, cnt_f/cnt_a*100))
            print('-'*16,'-------', '-'*16)
        return old_params
    #Method for saving fisher. mainly for debugging and testing purposes
    def save_fisher(self, net, DataLoader):
        net.apply(self.filter_bn_on)
        fisher = self.cal_fisher(net, DataLoader)
        net.apply(self.filter_bn_on)
        return fisher
    #call after making new new_model
    def forward(self, net, fisher, task_num, mode = 'channel',see_cnt = True):
        old_params = self.find_old_params(net, fisher)
        new_params = self.pruned_weights(old_params, fisher, task_num, mode, see_cnt)
        self.pruned_dict = {}
        for key in new_params.keys():
            self.pruned_dict[key] = (new_params[key]==0.)
        return new_params