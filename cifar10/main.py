import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import PIL
import torchvision
import torchvision.transforms as transforms
torch.nn.functional.cross_entropy()
import os
import argparse

from models import *
from utils import progress_bar
from termcolor import colored
import torch.nn.init as init

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
<<<<<<< HEAD
batch_size = 1000
=======
batch_size = 300
>>>>>>> 98e355a0a892c3427ccb37005300b0b55543778c
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    #transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    # transforms.RandomAffine(0, shear=5, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=15)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=15)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


# Model
print('==> Building model..')
<<<<<<< HEAD
# net = EfficientNetB0()
net = ResNet152()
=======
#net = EfficientNetB0()
# net = torchvision.models.resnext101_32x8d()
# net = ResNeXt29_2x64d()
# net = ResNet152()
# net = ResNeXt29_32x8d()
net = DenseNet40()
# net = resnext101_32x16d_wsl()
# net.apply(init_weights)

# net.fc = nn.Linear(2048,10)

>>>>>>> 98e355a0a892c3427ccb37005300b0b55543778c
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/t29_ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, nesterov=True)


scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
milestone = np.arange(250,350,1)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,190,250,300], gamma=0.1)


# Training
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

        #l1 loss
        # reg_loss = 0
        # for param in net.parameters():
        #     reg_loss += param.norm(p=1)
        # factor = 0.001
        # loss += factor * reg_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %4.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    scheduler[0].step()
    # scheduler[1].step()
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
<<<<<<< HEAD
        torch.save(state, './checkpoint/t152.pth')
=======

        torch.save(state, './checkpoint/tDN40.pth')
>>>>>>> 98e355a0a892c3427ccb37005300b0b55543778c
        best_acc = acc
        
    return best_acc

<<<<<<< HEAD
end_epoch = 400
=======
def kd_criterion(o_student, o_teacher, labels, T=3, w=0.8):

    KD_loss = nn.KLDivLoss()(F.log_softmax(o_student / T, dim=1),
                             F.softmax(o_teacher / T, dim=1)) * (w * T * T) + \
              F.cross_entropy(o_student, labels) * (1. - w)

    return KD_loss

end_epoch = 1500
>>>>>>> 98e355a0a892c3427ccb37005300b0b55543778c
epoch_tmp = 0
acc_tmp = 0
train_accs = []
test_accs = []
epoch_axis = range(end_epoch)
for epoch in range(start_epoch+end_epoch):
    train_acc = train(epoch, scheduler)
    test_acc = test(epoch)

    train_accs.append(train_acc)
    test_accs.append(test_acc)

    if acc_tmp < test_acc:
        acc_tmp = test_acc
        epoch_tmp = epoch

print('=============================')
print('best accuracy:', test_acc)
print('best epoch:', epoch_tmp)


