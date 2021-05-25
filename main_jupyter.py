# %load main_jupyter.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from modules.utils import plot_images, plot_classes

# Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py
# script
from model import Net
from networks import IDSIANetwork, GeneralNetwork

# Data Initialization and Loading
# data.py in the same folder
from data import initialize_data, data_transforms, val_data_transforms

args = argparse.Namespace()
args.data = 'data'
args.batch_size = 64
args.epochs = 25
args.lr = 0.0001
args.momentum = 0.5
args.seed = 1
args.weight = 0
args.log_interval = 10
args.cnn = None
args.locnet = '200, 300, 200'
args.locnet3 = '150, 150, 150'
args.st = True
args.save_loc = '/scratch/as10656'
args.model = '/scratch/as10656/model_best_st.pth'

torch.manual_seed(args.seed)

# extracts the zip files, makes a validation set
initialize_data(args.data)
train_dataset = datasets.ImageFolder(args.data + '/train_images',
                                     transform=data_transforms)
val_dataset = datasets.ImageFolder(args.data + '/val_images',
                                   transform=val_data_transforms)


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


weights = make_weights_for_balanced_classes(train_dataset.imgs,
                                            len(train_dataset.classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=1,
                                           sampler=sampler)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=1)


plot_images(train_loader)
plot_images(val_loader)
plot_classes(val_loader)

cuda_available = torch.cuda.is_available()

best_acc = 0
flag = 1
start_epoch = 1

if args.model:
    state = torch.load(args.model)

    if "args" in state:
        flag = 0
        model = IDSIANetwork(state['args'])
        model.cuda()
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
