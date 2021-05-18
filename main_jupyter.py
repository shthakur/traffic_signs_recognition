from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from modules.utils import plot_images

# Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py
# script
from model import Net
from networks import IDSIANetwork, GeneralNetwork

# Data Initialization and Loading
# data.py in the same folder
from data import initialize_data, data_transforms


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


args = {
    'data': 'data',
    'batch_size': 64,
    'epochs': 10,
    'lr': 0.001,
    'momentum': 0.5,
    'seed': 1,
    'weight': 0,
    'log_interval': 10,
    'cnn': None,
    'locnet': '200, 300, 200',
    'locnet3': '150, 150, 150',
    'st': True,
    'save_loc': '/scratch/as10656',
    'model': None
}

args = DotDict(args)
torch.manual_seed(args.seed)

initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)
# plot_images(train_loader)

cuda_available = torch.cuda.is_available()

best_acc = 0
flag = 1
start_epoch = 1

if args.model:
    state = torch.load(args.model)

    if "args" in state:
        flag = 0
        args = state['args']
        model = IDSIANetwork(args)
        model.load_state_dict(state['state_dict'])

        # For Adam specifically use lr = 0.001
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight)
        optimizer.load_state_dict(state['optimizer'])
        best_acc = state['best_prec']
        start_epoch = state['epoch'] + 1
if flag == 1:
    model = IDSIANetwork(args)
    if cuda_available:
        model = model.cuda()
    if args.model:
        model.load_state_dict(torch.load(args.model))
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        if cuda_available:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def validation(loader, loader_type):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in loader:
        data, target = Variable(data, volatile=True), Variable(target)

        if cuda_available:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(loader.dataset)
    print('\n' + loader_type + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

    return 100 * correct / len(loader.dataset)


def save_model(model_file, model, epoch, optimizer, best_acc):
    save = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_prec': best_acc,
        'epoch': epoch,
        'args': args
    }

    torch.save(save, model_file)


for epoch in range(start_epoch, args.epochs + 1):
    train(epoch)
    validation(train_loader, 'Train')
    val_acc = validation(val_loader, 'Validation')
    if val_acc > best_acc:
        best_acc = val_acc
        model_file = args.save_loc + '/model_best.pth'

        if args.st:
            model_file = args.save_loc + '/model_best_st.pth'

        save_model(model_file, model, epoch, optimizer, best_acc)
        print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
