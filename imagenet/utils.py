import torch
import os
import random
import shutil
from termcolor import colored
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS, default_loader
import globals

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()

def bn_finetune(m):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = max(1-10/(globals.idx+1),0.9)
def gaussian_noise(input, mean, stddev, alpha=0.8):
    for idx, batch in enumerate(input):
        p = random.random()
        if p < alpha:
            noise = batch.data.new(batch.size()).normal_(mean, stddev)
            input[idx] += noise
    return input

def save_checkpoint(state, is_best, teacher_name, student_name, save_path='', w='', acc=0):
    pid = os.getpid()

    if not os.path.isdir('weights'):
        os.mkdir('weights')
    if not os.path.isdir('{}'.format(save_path)):
        os.mkdir('{}'.format(save_path))
    torch.save(state, os.path.join(save_path, 'checkpoint{}_{}_{}_{}.pth.tar'.format(student_name,teacher_name,w,pid)))
    if is_best:
        print(colored('saving best...','red'))
        filename = os.path.join(save_path, 'checkpoint{}_{}_{}_{}.pth.tar'.format(student_name,teacher_name,w,pid))
        shutil.copyfile(filename, os.path.join(save_path, 'model_best:{}_{}_{:.2f}_{}.pth.tar'.format(student_name,teacher_name,acc,pid)))

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class ImageFolder_iid(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(ImageFolder_iid, self).__init__(root, loader, IMG_EXTENSIONS
        if is_valid_file is None else None,
                                              transform=transform,
                                              target_transform=target_transform,
                                              is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.img_to_idx = {i[0]: idx for idx, i in enumerate(self.imgs)}

    def __getitem__(self, index):
        path, target = self.samples[index]
        idx = self.img_to_idx[path]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, idx
