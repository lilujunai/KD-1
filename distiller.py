import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy
from utils import AverageMeter, accuracy
import math
import time

def train_with_overhaul(train_loader, d_net, optimizer, criterion_CE, epoch, args):
    d_net.train()
    d_net.module.s_net.train()
    d_net.module.t_net.train()

    train_loss = AverageMeter('Loss',':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    for i, (inputs, targets, idx) in enumerate(train_loader):
        targets = targets.cuda()
        batch_size = inputs.shape[0]
        outputs, loss_distill = d_net(inputs)
        loss_CE = criterion_CE(outputs, targets)
        loss = loss_CE + loss_distill.sum() / batch_size / 10000

        err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))

        train_loss.update(loss.item(), batch_size)
        top1.update(err1.item(), batch_size)
        top5.update(err5.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Train with distillation: [Epoch %d/%d][Batch %d/%d]\t Loss %.3f, Top 1-error %.3f, Top 5-error %.3f' %
                  (epoch, args.epochs, i, len(train_loader), train_loss.avg, top1.avg, top5.avg))

def validate_overhaul(val_loader, model, criterion_CE, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, idx) in enumerate(val_loader):
        target = target.cuda()

        # for PyTorch 0.4.x, volatile=True is replaced by with torch.no.grad(), so uncomment the followings:
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            output = model(input_var)
            loss = criterion_CE(output, target_var)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test (on val set): [Epoch {0}/{1}][Batch {2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                   epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'
          .format(epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg


def distillation_loss(source, target, margin):
    loss = ((source - margin)**2 * ((source > margin) & (target <= margin)).float() +
            (source - target)**2 * ((source > target) & (target > margin) & (target <= 0)).float() +
            (source - target)**2 * (target > 0).float())
    return torch.abs(loss).sum()

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)

class Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])
        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net
        # x = torch.rand((1,3,224,224))
        # t_feats, t_out = self.t_net.extract_feature(x, preReLU=True)
        # s_feats, s_out = self.s_net.extract_feature(x)
        # for i in range(4):
        #     print('s_feats:', s_feats[i].shape)
        #     print('t_feats:', t_feats[i].shape)
    def forward(self, x):
        with torch.no_grad():
            t_feats, t_out = self.t_net.extract_feature(x, preReLU=True)
        s_feats, s_out = self.s_net.extract_feature(x)
        feat_num = len(t_feats)

        loss_distill = 0
        for i in range(feat_num):
            s_feats[i] = self.Connectors[i](s_feats[i])
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
                            / 2 ** (feat_num - i - 1)

        return s_out, loss_distill