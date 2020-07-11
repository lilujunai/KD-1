import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from autoaugment import CIFAR10Policy
from models import *
import matplotlib.pyplot as plt
# from efficientnet.model import EfficientNet
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import progress_bar
from termcolor import colored

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--kd', action='store_true')
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b0')
parser.add_argument('-b', '--batch_size', default=256, type=int)
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
    root='/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=15)

testset = torchvision.datasets.CIFAR10(
    root='/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=15)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.kd:
    net = EfficientNetB0()
    t_net = ResNet152()

    net = net.to(device)
    t_net = t_net.to(device)
    if device == 'cuda':
        t_net = torch.nn.DataParallel(t_net)
        cudnn.benchmark = True

    checkpoint = torch.load('./t_ckpt.pth')
    t_net.load_state_dict(checkpoint['net'])

else:
    if args.arch == 'resnet152':
        net = ResNet152()
    elif args.arch == 'efficientnet-b0':
        net = EfficientNetB0()

net_name = net.__class__.__name__
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}.pth'.format(net_name))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * int(50000 / args.batch_size), eta_min=0,
                                      last_epoch=-1)
# milestone = np.arange(250,350,1)
# scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,190,250,300], gamma=0.1)


# Training
if args.kd:
    def train(epoch, scheduler):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            o_student = net(inputs)
            with torch.no_grad():
                o_teacher = t_net(inputs)
            loss = kd_criterion(o_student, o_teacher, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = o_student.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %4.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        scheduler.step()
        return 100. * correct / total
else:
    def train(epoch, scheduler):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %4.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        scheduler.step()
        return 100.*correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print(colored('Saving..','red'), end='')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        torch.save(state, './checkpoint/{}.pth'.format(net_name))
        best_acc = acc
        
    return best_acc

def kd_criterion(o_student, o_teacher, labels, T=3, w=0.8):

    KD_loss = nn.KLDivLoss()(F.log_softmax(o_student / T, dim=1),
                             F.softmax(o_teacher / T, dim=1)) * (w * T * T) + \
              F.cross_entropy(o_student, labels) * (1. - w)

    return KD_loss
if __name__ == '__main__':
    end_epoch = args.epochs
    epoch_tmp = 0
    acc_tmp = 0
    train_accs = []
    test_accs = []
    for epoch in range(start_epoch+end_epoch):
        train_acc = train(epoch, scheduler)
        test_acc = test(epoch)

        if acc_tmp < test_acc:
            acc_tmp = test_acc
            epoch_tmp = epoch
        train_accs.append(train_acc)
        test_accs.append(test_accs)
    print('=============================')
    print('best accuracy:', test_acc)
    print('best epoch:', epoch_tmp)
    state = {
                'net': net.state_dict(),
                'acc': acc_tmp,
                'epoch': epoch_tmp,
            }
    epochs = list(range(end_epoch))
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.savefig('accs.png')
    torch.save(state, './checkpoint/{}:{:.2f}.pth'.format(net_name,acc_tmp))
