import argparse
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import transforms
from utils import ImageFolder_iid, save_checkpoint
from efficientnet.model import EfficientNet
from scipy.spatial import distance
import torch.nn as nn
import torchvision
from autoaugment import ImageNetPolicy
import torch
import numpy as np
from size_estimator import SizeEstimator
from run import train, validate

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 pruning')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--l2', default=1., type=float, help='l2 norm pruning (1 = no pruning)')
parser.add_argument('--dist', default=0.9, type=float, help='median filter pruning (0 = no pruning)')
parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('--pth_path', default='./checkpoint/EfficientNet.pth', type=str)
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=2000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_path', default='', type=str)
args = parser.parse_args()

traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

train_dataset = ImageFolder_iid(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        ImageNetPolicy(),
        transforms.ToTensor(),
        normalize,
    ]))

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
else:
    train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    ImageFolder_iid(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

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
        self.layers_not_interested_name = ['bn']
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
                print(index, name)
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
            print('pruning idx by l2 norm: ', filter_small_index)
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
            print('pruning idx by dist: ', similar_index_for_filter)
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
        print('look up table')
        for i, (n, p) in enumerate(self.model.named_parameters()):
            look_up_table[n] = i
        return look_up_table

class KernelGarbageCollector:
    '''
    >>>GC = KernelGarbageCollector(model, pruned_kernel_idx, model_kernel_length, look_up_table)
    >>>GC.make_new_layer()
    >>>GC.cop y_unpruned_layers()
    >>>GC.overwrite_unpruned_layers()
    '''
    def __init__(self, model, pruned_kernel_idx, model_kernel_length, look_up_table):
        self.model = model
        self.pruned_kernel_idx = pruned_kernel_idx
        self.model_kernel_length = model_kernel_length
        self.look_up_table = look_up_table
        self.new_modules = {}

    def _is_module_pass(self, module_name):
        modules_in_eff = ['conv', 'bn', 'se']

        modules_to_pass = ['padding', 'drop', 'swish']
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
                    if 'depthwise' in name:
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
                previous_channel = output_channel

    def copy_unpruned_layers(self):
        previous_pruned_channel = torch.range(0, 2, dtype=int)
        for index, (name, module) in enumerate(self.model.named_parameters()):
            if '.weight' in name:
                pre_name = name[:-7]  # delete .weight in string
            elif '.bias' in name:
                pre_name = name[:-5]  # delete .bias in string
            print(name)

            if 'bn' in name:
                for i, idx in enumerate(unpruned_layers_idx):
                    #                     print('bn', i)
                    if 'weight' in name:
                        self.new_modules[pre_name].weight[i] = module[idx]
                    elif 'bias' in name:
                        self.new_modules[pre_name].bias[i] = module[idx]

            elif 'conv' in name or 'se' in name:
                if 'depth' in name:
                    unpruned_layers_idx = set(range(self.model_kernel_length[index])) - set(
                        self.pruned_kernel_idx[index])
                    unpruned_layers_idx = list(unpruned_layers_idx)

                    for i, idx in enumerate(unpruned_layers_idx):
                        self.new_modules[pre_name].weight[i] = module[idx]
                    previous_pruned_channel = torch.tensor(unpruned_layers_idx)

                else:
                    if 'weight' in name:
                        unpruned_layers_idx = set(range(self.model_kernel_length[index])) - set(
                            self.pruned_kernel_idx[index])
                        unpruned_layers_idx = list(unpruned_layers_idx)

                        for i, idx in enumerate(unpruned_layers_idx):
                            self.new_modules[pre_name].weight[i] = module[idx].index_select(dim=0,
                                                                                            index=previous_pruned_channel)
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

if __name__ == '__main__':
    model = EfficientNet.from_name('efficientnet-b0')
    model = torch.nn.DataParallel(model).cuda()

    loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.load(args.pth_path, map_location=loc)
    model.load_state_dict(checkpoint['state_dict'])

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay,
                                  amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * int(1281167 / args.batch_size), eta_min=0,
                                  last_epoch=-1)

    se1 = SizeEstimator(model, input_size=(1, 3, 224, 224))
    param_size1 = se1.get_parameter_sizes()
    act_size1 = se1.get_output_sizes()
    size1 = param_size1 + act_size1

    pr = Prune(model, args.l2, args.dist)
    look_up_table = pr.look_up_table()
    pr.init()
    pr.do_mask()
    pruned_kernel_idx = pr.get_pruned_kernel_idx()
    model_kernel_length = pr.get_model_kernel_length()
    pr.if_zero()

    se2 = SizeEstimator(model, input_size=(1, 3, 224, 224))
    param_size2 = se2.get_parameter_sizes()
    act_size2 = se2.get_output_sizes()
    size2 = param_size2 + act_size2
    pruned_ratio = (1 - (size1 / size2))

    print('pruned ratio:', pruned_ratio)
    print('from:', size1, 'to:', size2)


    # model = pr.model
    # model = torch.nn.DataParallel(model).cuda()
    best_acc1 = 0
    for epoch in range(0, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)
        acc1 = validate(val_loader, model, criterion, args)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        model_name = model.module.__class__.__name__
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, teacher_name='', student_name=model_name, save_path=args.save_path, acc=acc1)

        scheduler.step()