import torch
import os
import random
import shutil
from torch.autograd import Variable
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS, default_loader

def gaussian_noise(input, mean, stddev, alpha=0.8):
    for idx, batch in enumerate(input):
        p = random.random()
        if p < alpha:
            noise = batch.data.new(batch.size()).normal_(mean, stddev)
            input[idx] += noise
    return input

def save_checkpoint(state, is_best, save_path=''):
    torch.save(state, os.path.join(save_path, 'checkpoint.pth.tar'))
    if is_best:
        filename = os.path.join(save_path, 'checkpoint.pth.tar')
        shutil.copyfile(filename, os.path.join(save_path, 'model_best.pth.tar'))

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