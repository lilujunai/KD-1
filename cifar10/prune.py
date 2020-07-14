import time

from scipy.spatial import distance
import torch.nn as nn
import torch
import numpy as np
from models import *
import os
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from autoaugment import CIFAR10Policy
import argparse
from size_estimator import SizeEstimator
from utils import progress_bar
from torch.optim.lr_scheduler import CosineAnnealingLR
from termcolor import colored

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 pruning')
parser.add_argument('--l2', default=0.98, type=float, help='l2 norm pruning (1 = no pruning)')
parser.add_argument('--dist', default=0.05, type=float, help='median filter pruning (0 = no pruning)')
parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('--pth_path', default='./checkpoint/EfficientNet:93.83.pth', type=str)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

batch_size = args.batch_size
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    CIFAR10Policy(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='/data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=15)

testset = torchvision.datasets.CIFAR10(
    root='/data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=15)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()


# def train(epoch, scheduler):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         pr.do_grad_mask()
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()
#
#         progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %4.3f%% (%d/%d)'
#                      % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
#     scheduler.step()
#     return 100. * correct / total

# def test(epoch):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
#
#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#
#             progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                          % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
#
#     # Save checkpoint.
#     acc = 100. * correct / total
#     if acc > best_acc:
#         print(colored('Saving..', 'red'), end='')
#         state = {
#             'net': net.model.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#
#         torch.save(state, './checkpoint/{}.pth'.format(net_name))
#         best_acc = acc
#
#     return best_acc

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def train(train_loader, model, criterion, optimizer, epoch, m):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # Mask grad for iteration
        m.do_grad_mask()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1000 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % 1000 == 0:
            print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                           error1=100 - top1.avg))

    return top1.avg

class Prune:
    '''
    >>>pr = Prune(model, 1, 0.1)
    >>>look_up_table = pr.look_up_table()
    >>>pr.init()
    >>>pr.do_mask()
    >>>pruned_kernel_idx = pr.get_pruned_kernel_idx()
    >>>model_kernel_length = pr.get_model_kernel_length()
    >>>pr.if_zero()
    '''
    def __init__(self, model, l1_prune_ratio, median_pruning_ratio):
        self.model = model
        self.compress_rate = {}
        self.distance_rate = {}
        self.mask_index = []
        self.mat = {}
        self.model_length = {}  # parameter length of each kernel
        self.model_size = {}
        self.similar_matrix = {}
        self.pruning_params_name = ['conv', 'se']
        self.skip_connect_name = 'project'
        self.layers_not_interested_name = ['bn', 'se.bias']
        self.rate_norm_per_layer = l1_prune_ratio
        self.rate_dist_per_layer = median_pruning_ratio
        self.model_idx_dict = self.get_model_idx_dict()
        self.layer_length = self.get_layer_length()
        self.skip_connection_idx = self.get_skip_connections()
        self.not_interested_idx = self.get_idx_not_interested()
        self._count_pruned_idx = 0

    def init(self, prune_skip_connect=False):
        self.get_model_length()
        print('initializing masks...')
        self.init_pruning_rate()
        print('idx to prune (add layer name to pruning_params_name to prune more)', self.mask_index)
        self._get_maskmatrix()

    def get_idx_not_interested(self):
        not_interested_layers_idx = []
        for index, (name, item) in enumerate(self.model.named_parameters()):
            if self.does_string_exist(self.layers_not_interested_name, name):
                not_interested_layers_idx.append(index)
        print('not interested idx:', not_interested_layers_idx)
        return not_interested_layers_idx

    # writes model's parameter number on self.model_length
    def get_model_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]
        print('model length:', self.model_length)

    # returns model's kernel number for each layers
    def get_model_kernel_length(self):
        model_kernel_length = {}
        for index, item in enumerate(self.model.parameters()):
            model_kernel_length[index] = item.size()[0]
        return model_kernel_length

    def get_layer_length(self):
        length = 0
        for index, item in enumerate(self.model.parameters()):
            length = index
        print('number of layers in model: ', length)
        return length

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index].cpu()
                item.data = b.view(self.model_size[index])

                a = item.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.data = b.view(self.model_size[index])

    def do_grad_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.grad.data.view(self.model_length[index]).cuda()
                # reverse the mask of model
                # b = a * (1 - self.mat[index])
                b = a * self.mat[index].cuda()
                b = b * self.similar_matrix[index].cuda()
                item.grad.data = b.view(self.model_size[index])

    def get_skip_connections(self):
        skip_connection_idx = []
        for index, (name, item) in enumerate(self.model.named_parameters()):
            if self.skip_connect_name in name:
                skip_connection_idx.append(index)
                skip_connection_idx.append(index + 1)  # consequtive bn.weight after conv
                skip_connection_idx.append(index + 2)  # consequtive bn.bias after conv
        print('skip connections in indexes: ', skip_connection_idx)
        return skip_connection_idx

    def get_model_idx_dict(self):
        index_dict = {}
        for index, (name, item) in enumerate(self.model.named_parameters()):
            index_dict[index] = name
        return index_dict

    def does_string_exist(self, str1, str2):
        for name in str1:
            if name in str2:
                return True
        return False

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def _get_maskmatrix(self):
        for index, (name, item) in enumerate(self.model.named_parameters()):
            if index in self.mask_index:
                # mask for norm criterion
                # print(index, name)
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = torch.FloatTensor(self.mat[index])

                # mask for distance criterion
                self.similar_matrix[index] = self.get_filter_similar(item.data, self.compress_rate[index],
                                                                     self.distance_rate[index],
                                                                     self.model_length[index])
                self.similar_matrix[index] = torch.FloatTensor(self.similar_matrix[index])

    def init_pruning_rate(self, prune_skip_connect=False):
        # initialize pruning rate
        pruning_idx = []  # just for debugging
        self.mask_index = [x for x in range(self.layer_length)]
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
            self.distance_rate[index] = 1

        for key in range(self.layer_length):
            if self.does_string_exist(self.pruning_params_name, self.model_idx_dict[key]):
                self.compress_rate[key] = self.rate_norm_per_layer
                self.distance_rate[key] = self.rate_dist_per_layer
                pruning_idx.append(key)

        # no pruning on not interest layers
        for x in self.not_interested_idx:
            self.compress_rate[x] = 1
            self.distance_rate[x] = 1
            self.mask_index.remove(x)
        # no pruning on skip connection
        if not prune_skip_connect:
            for x in self.skip_connection_idx:
                try:
                    self.compress_rate[x] = 1
                    self.distance_rate[x] = 1
                    self.mask_index.remove(x)
                except:
                    print('idx:{} is already deleted from unwanted layers'.format(x))

        print('prune rate (l2) for all index: (1 for no pruning)', self.compress_rate)
        print('prune rate (dist) for all index: (1 for no pruning)', self.distance_rate)

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:  # if kernel
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0
        #         print("filter codebook done")
        else:
            pass
        return codebook

    def get_filter_similar(self, weight_torch, compress_rate, distance_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:  # output chn, input or kernel chn, kenrelsize, kenrelsize
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))  # filters pruned by norm
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)  # filters to prune here
            weight_vec = weight_torch.view(weight_torch.size()[0], -1).cuda()  # numoffilters, weights
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            # norm based pruning (not interested)
            norm2 = torch.norm(weight_vec, 2, 1)  # norm on each kernel (ch,size,size)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []  # not used
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]  # not used [:0]
            # print('pruning idx by l2 norm: ', filter_small_index)
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            # median based pruning (interested)
            # distance using numpy function
            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()  # select unremoved weights
            # for euclidean distance
            similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
            # for cos similarity
            # similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix),
                                 axis=0)  # considering all other kernels. small = similar to others
            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]  # not used
            similar_small_index = similar_sum.argsort()[:similar_pruned_num]  # small distance
            similar_index_for_filter = [filter_large_index[i] for i in
                                        similar_small_index]  # using filter_large_index[i] because of torch.index_select
            # print('pruning idx by dist: ', similar_index_for_filter)
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
        #         print("similar index done")
        else:
            pass
        return codebook

    def get_pruned_kernel_idx(self):
        pruned_kernel_idx = {}
        for index, (name, item) in enumerate(self.model.named_parameters()):
            if index in self.not_interested_idx:
                pruned_kernel_idx[index] = []
                continue
            if index in self.skip_connection_idx:
                pruned_kernel_idx[index] = []
                continue
            filter_pruned_num = int(item.data.size()[0] * (1 - self.compress_rate[index]))
            similar_pruned_num = int(item.data.size()[0] * self.distance_rate[index])
            weight_vec = item.data.view(item.data.size()[0], -1).cuda()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = []
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()  # select unremoved weights
            similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            pruned_kernel_idx[index] = filter_small_index.tolist() + similar_index_for_filter
        return pruned_kernel_idx

    def if_zero(self):
        for index, (name, item) in enumerate(self.model.named_parameters()):
            if index in self.model_length:
                # if index in [x for x in range(args.layer_begin, args.layer_end + 1, args.layer_inter)]:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()
                print(name)
                print("layer: %d, number of nonzero weight is %d, zero is %d" % (
                    index, np.count_nonzero(b), len(b) - np.count_nonzero(b)))

    def look_up_table(self):
        look_up_table = {}
        # print('look up table')
        for i, (n, p) in enumerate(self.model.named_parameters()):
            look_up_table[n] = i
        return look_up_table

class KernelGarbageCollector:
    '''
    >>>GC = KernelGarbageCollector(model, pruned_kernel_idx, model_kernel_length, look_up_table)
    >>>GC.make_new_layer()
    >>>GC.copy_unpruned_layers()
    >>>GC.overwrite_unpruned_layers()
    '''
    def __init__(self, model, pruned_kernel_idx, model_kernel_length, look_up_table):
        self.model = model
        self.pruned_kernel_idx = pruned_kernel_idx
        self.model_kernel_length = model_kernel_length
        self.look_up_table = look_up_table
        self.new_modules = {}

    def _is_module_pass(self, module_name):
        modules_in_eff = ['conv', 'bn', 'se1','se2','linear']

        modules_to_pass = ['padding', 'drop', 'relu']
        for module in modules_to_pass:
            if module in module_name:
                return True

        for module in modules_in_eff:
            if module in module_name:
                return False
        return True

    def get_module_idx(self, name):
        name = name + '.weight'
        index = self.look_up_table[name]
        return index

    def make_new_layer(self):
        previous_channel = 3  # channel number for color (rgb)
        for index, (name, module) in enumerate(self.model.named_modules()):
            if not self._is_module_pass(name):
                #                     print(name)
                index = self.get_module_idx(name)
                #                     print(index, module)

                if 'conv' in name or 'se' in name:
                    output_channel = self.model_kernel_length[index] - len(self.pruned_kernel_idx[index])
                    if 'conv2' in name: # conv2 = depthwise conv
                        module.groups = output_channel
                    tmp_conv = nn.Conv2d(previous_channel,
                                         output_channel,
                                         kernel_size=module.kernel_size,
                                         stride=module.stride,
                                         padding=module.padding,
                                         dilation=module.dilation,
                                         groups=module.groups,
                                         bias=False,
                                         padding_mode=module.padding_mode)
                    #                         print(tmp_conv)
                    self.new_modules[name] = tmp_conv
                if 'bn' in name:
                    tmp_bn = nn.BatchNorm2d(output_channel,
                                            eps=module.eps,
                                            momentum=module.momentum,
                                            affine=module.affine,
                                            track_running_stats=module.track_running_stats)
                    self.new_modules[name] = tmp_bn
                if 'linear' in name:
                    tmp_linear = nn.Linear(previous_channel, module.out_features, bias=True)
                    self.new_modules[name] = tmp_linear
                previous_channel = output_channel
                # print(name, self.new_modules[name])

    def copy_unpruned_layers(self):
        previous_pruned_channel = torch.range(0, 2, dtype=int)
        for index, (name, module) in enumerate(self.model.named_parameters()):
            if '.weight' in name:
                pre_name = name[:-7]  # delete .weight in string
            elif '.bias' in name:
                pre_name = name[:-5]  # delete .bias in string


            if 'bn' in name:
                for i, idx in enumerate(unpruned_layers_idx):
                    #                     print('bn', i)
                    if 'weight' in name:
                        self.new_modules[pre_name].weight[i] = module[idx]
                        # self.new_modules[pre_name].weight[i].grad_fn = module[idx].grad_fn
                    elif 'bias' in name:
                        self.new_modules[pre_name].bias[i] = module[idx]

            elif 'conv' in name or 'se' in name:
                if 'conv2' in name:  # conv2 = depthwise conv
                    unpruned_layers_idx = set(range(self.model_kernel_length[index])) - set(
                        self.pruned_kernel_idx[index])
                    unpruned_layers_idx = list(unpruned_layers_idx)

                    for i, idx in enumerate(unpruned_layers_idx):
                        self.new_modules[pre_name].weight[i] = module[idx].double().requires_grad_()
                        print(self.new_modules[pre_name].weight[i].grad_fn)
                    previous_pruned_channel = torch.tensor(unpruned_layers_idx)

                else:
                    if 'weight' in name:

                        unpruned_layers_idx = set(range(self.model_kernel_length[index])) - set(
                            self.pruned_kernel_idx[index])
                        unpruned_layers_idx = list(unpruned_layers_idx)

                        for i, idx in enumerate(unpruned_layers_idx):
                            self.new_modules[pre_name].weight[i] = module[idx].index_select(dim=0,
                                                                                            index=previous_pruned_channel).double().requires_grad_()

                        # print(name, self.new_modules[pre_name])
                        previous_pruned_channel = torch.tensor(unpruned_layers_idx, dtype=int)
                    elif 'bias' in name:
                        pass

    def copy_unpruned_layers_nyw(self):
        previous_pruned_channel = torch.range(0, 2, dtype=int)
        for index, (name, module) in enumerate(self.model.named_parameters()):
            print(name)
            continue
            if 'bn' in name:
                for i, idx in enumerate(unpruned_layers_idx):
                    #                     print('bn', i)
                    if 'weight' in name:
                        self.new_modules[pre_name].weight[i] = module[idx]
                        # self.new_modules[pre_name].weight[i].grad_fn = module[idx].grad_fn
                    elif 'bias' in name:
                        self.new_modules[pre_name].bias[i] = module[idx]

            elif 'conv' in name or 'se' in name:
                if 'conv2' in name:  # conv2 = depthwise conv
                    unpruned_layers_idx = set(range(self.model_kernel_length[index])) - set(
                        self.pruned_kernel_idx[index])
                    unpruned_layers_idx = list(unpruned_layers_idx)

                    for i, idx in enumerate(unpruned_layers_idx):
                        self.new_modules[pre_name].weight[i] = module[idx].double().requires_grad_()
                        print(self.new_modules[pre_name].weight[i].grad_fn)
                    previous_pruned_channel = torch.tensor(unpruned_layers_idx)

                else:
                    if 'weight' in name:

                        unpruned_layers_idx = set(range(self.model_kernel_length[index])) - set(
                            self.pruned_kernel_idx[index])
                        unpruned_layers_idx = list(unpruned_layers_idx)

                        for i, idx in enumerate(unpruned_layers_idx):
                            self.new_modules[pre_name].weight[i] = module[idx].index_select(dim=0,
                                                                                            index=previous_pruned_channel).double().requires_grad_()

                        # print(name, self.new_modules[pre_name])
                        previous_pruned_channel = torch.tensor(unpruned_layers_idx, dtype=int)
                    elif 'bias' in name:
                        pass


    def overwrite_unpruned_layers(self):
        for key in self.new_modules:
            keys = key.split('.')
            if len(keys) == 1:
                self.model._modules[keys[0]] = self.new_modules[key]
            elif len(keys) == 3:
                self.model._modules[keys[0]][int(keys[1])]._modules[keys[2]] = self.new_modules[key]
            elif len(keys) == 4:\
                self.model._modules[keys[0]][int(keys[1])]._modules[keys[2]]._modules[keys[3]] = self.new_modules[key]

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = EfficientNetB0()
    net_name = net.__class__.__name__
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    checkpoint = torch.load('{}'.format(args.pth_path))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = 0
    net = net.module
    net = net.module.cpu()

    se1 = SizeEstimator(net, input_size=(1, 3, 32, 32))
    param_size1 = se1.get_parameter_sizes()
    act_size1 = se1.get_output_sizes()
    size1 = param_size1 + act_size1

    pr = Prune(net, args.l2, args.dist)
    look_up_table = pr.look_up_table()
    pr.init()
    pr.do_mask()
    pruned_kernel_idx = pr.get_pruned_kernel_idx()
    model_kernel_length = pr.get_model_kernel_length()
    pr.if_zero()
    net = pr.model
    # GC = KernelGarbageCollector(net, pruned_kernel_idx, model_kernel_length, look_up_table)
    # GC.make_new_layer()
    # GC.copy_unpruned_layers_nyw()
    # GC.overwrite_unpruned_layers()

    # net = GC.model

    net = net.to(device)

    se2 = SizeEstimator(net, input_size=(1, 3, 32, 32))
    param_size2 = se2.get_parameter_sizes()
    act_size2 = se2.get_output_sizes()
    size2 = param_size2 + act_size2
    print(act_size2)

    pruned_ratio = ((size2/size1) * 100)

    print('pruned ratio:', pruned_ratio)
    print('from:', size1, 'to:', size2)
    print('=========================')

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                                  amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * int(50000 / args.batch_size), eta_min=0,
                                  last_epoch=-1)
    end_epoch = args.epochs
    epoch_tmp = 0
    acc_tmp = 0
    for epoch in range(start_epoch + end_epoch):
        train_acc = train(train_loader=trainloader, model=net, criterion=criterion, optimizer=optimizer, epoch=epoch, m=pr)
        test_acc = validate(val_loader=testloader, model=net, criterion=criterion)

        if acc_tmp < test_acc:
            acc_tmp = test_acc
            epoch_tmp = epoch
            best_state = {
              'net': net.state_dict(),
              'acc': acc_tmp,
              'epoch': epoch_tmp,
            }
            if acc_tmp > 90:
                torch.save(state, './checkpoint/{}:{:.2f}.pth'.format(net_name,acc_tmp))
    print('=============================')
    print('best accuracy:', test_acc)
    print('best epoch:', epoch_tmp)
    state = {
        'net': net.state_dict(),
        'acc': acc_tmp,
        'epoch': epoch_tmp,
    }
    pr.if_zero()
    torch.save(state, './checkpoint/{}.pth'.format(net_name))
