import argparse
import os
import random
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from cifar10_models import *
from termcolor import colored
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import ImageFolder_iid, save_checkpoint
from efficientnet.model import EfficientNet
from run import train_kd, validate_kd, train, validate, kd_criterion
import resnet
from distiller import train_with_overhaul, validate_overhaul, Distiller
from autoaugment import ImageNetPolicy

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
for i in range(8):
    model_names.append('efficientnet-b{}'.format(i))
model_names.append('resnext101_32x8d')
model_names.append('resnext101_32x16d')
model_names.append('DenseNet40')
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--T', type=float, help='temperature value for distillation')
parser.add_argument('--w', default=0.8, type=float, help='ratio for cross entropy loss and distillation loss')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                  metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=6e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=2000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--kd', action='store_true')
parser.add_argument('--overhaul', action='store_true')
parser.add_argument('--teacher_arch', default='resnet152',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet152)')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[150,250,350],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--save_path', default='', type=str)
parser.add_argument('--pth_path', default='./weights/resnet152_efficientnet-b0/model_best:EfficientNet_ResNet.pth.tar', type=str)
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')

best_acc1 = 0

def main():
    args = parser.parse_args()
    if (not args.kd) and args.overhaul:
        print('overhaul option should be given with kd option')
        return

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

# create model
#####################################################################################

    if args.pretrained:
        if args.arch.startswith('efficientnet-b'):
            print('=> using pre-trained {}'.format(args.arch))
            model = EfficientNet.from_pretrained(args.arch, advprop=args.advprop)

        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
    else:
        if args.arch.startswith('efficientnet-b'):
            print("=> creating model {}".format(args.arch))
            model = EfficientNet.from_name(args.arch)
        elif args.arch.startswith('Dense'):
            print("=> creating model {}".format(args.arch))
            model = DenseNet40()
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    # create teacher model
    if args.kd:
        print('=> loading teacher model')
        if args.teacher_arch.startswith('efficientnet-b'):
            teacher = EfficientNet.from_pretrained(args.teacher_arch)
            teacher.eval()
            print('=> {} loaded'.format(args.teacher_arch))

        elif args.teacher_arch.startswith('resnext101_32'):
            teacher = torch.hub.load('facebookresearch/WSL-Images', '{}_wsl'.format(args.teacher_arch))
            teacher.eval()
            print('=> {} loaded'.format(args.teacher_arch))
        elif args.overhaul:
            teacher = resnet.resnet152(pretrained=True)
        else:
            teacher = models.__dict__[args.teacher_arch](pretrained=True)
            teacher.eval()
            print('=> {} loaded'.format(args.teacher_arch))

        if args.overhaul:
            print('=> using overhaul distillation')
            d_net = Distiller(teacher, model)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            if args.kd:
                teacher = torch.nn.DataParallel(teacher).cuda()
                if args.overhaul:
                    d_net = torch.nn.DataParallel(d_net).cuda()

    if args.pretrained:
        if args.arch.startswith('efficientnet-b'):
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.pth_path, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'])
#####################################################################################


# define loss function (criterion) and optimizer, scheduler
#####################################################################################
    if args.kd:
        criterion = kd_criterion
        if args.overhaul:
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)


    if args.overhaul:
        optimizer = torch.optim.SGD(list(model.parameters()) + list(d_net.module.Connectors.parameters()), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)  # nesterov
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * int(1281167 / args.batch_size), eta_min=0,
                                      last_epoch=-1)
        args.lr = 0.048
        args.bs = 384
        optimizer = torch.optim.RMSprop(
            model.parameters, lr=args.lr, alpha=0.9, eps=.001,
            momentum=0.9, weight_decay=args.wd)

        from typing import Dict, Any
        class Scheduler:
            """ Parameter Scheduler Base Class
            A scheduler base class that can be used to schedule any optimizer parameter groups.
            Unlike the builtin PyTorch schedulers, this is intended to be consistently called
            * At the END of each epoch, before incrementing the epoch count, to calculate next epoch's value
            * At the END of each optimizer update, after incrementing the update count, to calculate next update's value
            The schedulers built on this should try to remain as stateless as possible (for simplicity).
            This family of schedulers is attempting to avoid the confusion of the meaning of 'last_epoch'
            and -1 values for special behaviour. All epoch and update counts must be tracked in the training
            code and explicitly passed in to the schedulers on the corresponding step or step_update call.
            Based on ideas from:
             * https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler
             * https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers
            """

            def __init__(self,
                         optimizer: torch.optim.Optimizer,
                         param_group_field: str,
                         noise_range_t=None,
                         noise_type='normal',
                         noise_pct=0.67,
                         noise_std=1.0,
                         noise_seed=None,
                         initialize: bool = True) -> None:
                self.optimizer = optimizer
                self.param_group_field = param_group_field
                self._initial_param_group_field = f"initial_{param_group_field}"
                if initialize:
                    for i, group in enumerate(self.optimizer.param_groups):
                        if param_group_field not in group:
                            raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                        group.setdefault(self._initial_param_group_field, group[param_group_field])
                else:
                    for i, group in enumerate(self.optimizer.param_groups):
                        if self._initial_param_group_field not in group:
                            raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
                self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
                self.metric = None  # any point to having this for all?
                self.noise_range_t = noise_range_t
                self.noise_pct = noise_pct
                self.noise_type = noise_type
                self.noise_std = noise_std
                self.noise_seed = noise_seed if noise_seed is not None else 42
                self.update_groups(self.base_values)

            def state_dict(self) -> Dict[str, Any]:
                return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

            def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
                self.__dict__.update(state_dict)

            def get_epoch_values(self, epoch: int):
                return None

            def get_update_values(self, num_updates: int):
                return None

            def step(self, epoch: int, metric: float = None) -> None:
                self.metric = metric
                values = self.get_epoch_values(epoch)
                if values is not None:
                    values = self._add_noise(values, epoch)
                    self.update_groups(values)

            def step_update(self, num_updates: int, metric: float = None):
                self.metric = metric
                values = self.get_update_values(num_updates)
                if values is not None:
                    values = self._add_noise(values, num_updates)
                    self.update_groups(values)

            def update_groups(self, values):
                if not isinstance(values, (list, tuple)):
                    values = [values] * len(self.optimizer.param_groups)
                for param_group, value in zip(self.optimizer.param_groups, values):
                    param_group[self.param_group_field] = value

            def _add_noise(self, lrs, t):
                if self.noise_range_t is not None:
                    if isinstance(self.noise_range_t, (list, tuple)):
                        apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
                    else:
                        apply_noise = t >= self.noise_range_t
                    if apply_noise:
                        g = torch.Generator()
                        g.manual_seed(self.noise_seed + t)
                        if self.noise_type == 'normal':
                            while True:
                                # resample if noise out of percent limit, brute force but shouldn't spin much
                                noise = torch.randn(1, generator=g).item()
                                if abs(noise) < self.noise_pct:
                                    break
                        else:
                            noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
                        lrs = [v + v * noise for v in lrs]
                return lrs
        class StepLRScheduler(Scheduler):
            """
            """

            def __init__(self,
                         optimizer: torch.optim.Optimizer,
                         decay_t: float,
                         decay_rate: float = 1.,
                         warmup_t=0,
                         warmup_lr_init=0,
                         t_in_epochs=True,
                         noise_range_t=None,
                         noise_pct=0.67,
                         noise_std=1.0,
                         noise_seed=42,
                         initialize=True,
                         ) -> None:
                super().__init__(
                    optimizer, param_group_field="lr",
                    noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
                    initialize=initialize)

                self.decay_t = decay_t
                self.decay_rate = decay_rate
                self.warmup_t = warmup_t
                self.warmup_lr_init = warmup_lr_init
                self.t_in_epochs = t_in_epochs
                if self.warmup_t:
                    self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
                    super().update_groups(self.warmup_lr_init)
                else:
                    self.warmup_steps = [1 for _ in self.base_values]

            def _get_lr(self, t):
                if t < self.warmup_t:
                    lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
                else:
                    lrs = [v * (self.decay_rate ** (t // self.decay_t)) for v in self.base_values]
                return lrs

            def get_epoch_values(self, epoch: int):
                if self.t_in_epochs:
                    return self._get_lr(epoch)
                else:
                    return None

            def get_update_values(self, num_updates: int):
                if not self.t_in_epochs:
                    return self._get_lr(num_updates)
                else:
                    return None

        scheduler = StepLRScheduler(
            optimizer,
            decay_t=2.4,
            decay_rate=0.97,
            warmup_lr_init=1e-6,
            warmup_t=3,
            noise_range_t=None,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )
    # scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
    # milestone = np.ceil(np.arange(0,300,2.4))

    # scheduler = MultiStepLR(optimizer, milestones=[30,60,90,120,150,180,210,240,270], gamma=0.1)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
#####################################################################################


# Data loading code
#####################################################################################
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
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
#####################################################################################

    if args.evaluate:
        validate(val_loader, model, criterion, args)

# Start training
#####################################################################################
    best_acc1 = 0
    teacher_name = ''
    student_name = ''
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        if args.kd:
            if args.overhaul:
                train_with_overhaul(train_loader, d_net, optimizer, criterion, epoch, args)
                acc1 = validate_overhaul(val_loader, model, criterion, epoch, args)
            else:
                train_kd(train_loader, teacher, model, criterion, optimizer, epoch, args)
                acc1 = validate_kd(val_loader, teacher, model, criterion, args)

                teacher_name = teacher.module.__class__.__name__

        else:
            student_name = model.module.__class__.__name__
            train(train_loader, model, criterion, optimizer, epoch, args)
            acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        #writer.add_scalars('acc1', acc1, epoch)
        is_best = acc1 > best_acc1
        if acc1 < 65:
            print(colored('not saving... accuracy smaller than 65',' green'))
            is_best = False
        best_acc1 = max(acc1, best_acc1)


        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, teacher_name=teacher_name, student_name=student_name, save_path=args.save_path, acc=acc1)

        scheduler.step()
#####################################################################################

if __name__ == '__main__':
    main()
