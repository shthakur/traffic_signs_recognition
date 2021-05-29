import torch
from networks import IDSIANetwork
from torch import optim
from torch.autograd import Variable
from torch import nn
from modules.EarlyStopping import EarlyStopping
from modules.utils import utils
from torch.utils.data import DataLoader

import time
import numpy as np


class Trainer:
    def __init__(self, params, train_data=None, val_data=None):
        self.params = params
        self.train_data = train_data
        self.val_data = val_data

        print("Creating dataloaders")
        self.cuda_available = torch.cuda.is_available()

        if self.train_data is not None:
            self.train_loader = DataLoader(dataset=self.train_data,
                                           shuffle=True,
                                           batch_size=params.batch_size,
                                           pin_memory=self.cuda_available)
        if self.val_data is not None:
            self.val_loader = DataLoader(dataset=self.val_data,
                                         shuffle=False,
                                         batch_size=params.batch_size,
                                         pin_memory=self.cuda_available)

        self.string_fixer = "=========="

    def load(self):
        self.model = IDSIANetwork(self.params)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                           self.model.parameters()),
                                    lr=self.params.lr)

        self.start_time = time.time()
        self.histories = {
            "train_loss": np.empty(0, dtype=np.float32),
            "train_acc": np.empty(0, dtype=np.float32),
            "val_loss": np.empty(0, dtype=np.float32),
            "val_acc": np.empty(0, dtype=np.float32)
        }

        # We minimize the cross entropy loss here
        self.early_stopping = EarlyStopping(
            self.model, self.optimizer, params=self.params,
            patience=self.params.patience, minimize=True)

        if self.params.resume:
            checkpoint = utils.load_checkpoint(self.params.resume)
            if checkpoint is not None:

                if "params" in checkpoint:
                    # To make sure model architecture remains same
                    self.params.locnet = checkpoint['params'].locnet
                    self.params.locnet2 = checkpoint['params'].locnet2
                    self.params.locnet3 = checkpoint['params'].locnet3
                    self.params.st = checkpoint['params'].st

                    self.model = IDSIANetwork(self.params)
                    self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                                       self.model.parameters()),
                                                lr=self.params.lr)

                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.histories.update(checkpoint)
                self.early_stopping.init_from_checkpoint(checkpoint)
                print("Loaded model, Best Loss: %.8f, Best Acc: %.2f" %
                      (checkpoint['best'], checkpoint['best_acc']))

        if self.cuda_available:
            self.model = self.model.cuda()

    def train(self):
        self.epochs = self.params.epochs

        criterion = nn.CrossEntropyLoss()
        start_epoch = 0

        self.model.train()
        print("Starting training")
        self.print_info()
        for epoch in range(start_epoch, self.params.epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images_batch = Variable(images)
                labels_batch = Variable(labels)

                if self.cuda_available:
                    images_batch = images_batch.cuda()
                    labels_batch = labels_batch.cuda(async=True)

                self.optimizer.zero_grad()
                output = self.model(images_batch)
                loss = criterion(output, labels_batch.long())
                loss.backward()
                self.optimizer.step()

                if self.params.extra_debug and (i + 1) % (self.params.batch_size * 4) == 0:
                    print(('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4},')
                          .format(epoch + 1,
                                  self.params.epochs,
                                  i + 1,
                                  len(self.train_loader),
                                  loss.data[0]))

            train_acc, train_loss = self.validate_model(self.train_loader, self.model)
            val_acc, val_loss = self.validate_model(self.val_loader, self.model)

            self.histories['train_loss'] = np.append(self.histories['train_loss'], [train_loss])
            self.histories['val_loss'] = np.append(self.histories['val_loss'], [val_loss])
            self.histories['val_acc'] = np.append(self.histories['val_acc'], [val_acc])
            self.histories['train_acc'] = np.append(self.histories['train_acc'], [train_acc])

            if not self.early_stopping(val_loss, val_acc, epoch, self.histories):
                self.print_train_info(epoch, train_acc, train_loss, val_acc, val_loss)
            else:
                print("Early stopping activated")
                print("Restoring earlier state and stopping")
                self.early_stopping.print_info()
                break

    def validate_model(self, loader, model):
        model.eval()
        correct = 0
        total = 0
        total_loss = 0

        for images, labels in loader:
            images_batch = Variable(images, volatile=True)
            labels_batch = Variable(labels.long())

            if self.cuda_available:
                images_batch = images_batch.cuda()
                labels_batch = labels_batch.cuda()

            output = model(images_batch)
            loss = nn.functional.cross_entropy(output, labels_batch.long(), size_average=False)
            total_loss += loss.data[0]
            total += len(labels_batch)

            if not self.cuda_available:
                correct += (labels_batch == output.max(1)[1]).data.cpu().numpy().sum()
            else:
                correct += (labels_batch == output.max(1)[1]).data.sum()
        model.train()

        average_loss = total_loss / total
        return correct / total * 100, average_loss

    def print_info(self):
        print(self.string_fixer + " Data " + self.string_fixer)
        print("Training set: %d examples" % (len(self.train_data)))
        print("Validation set: %d examples" % (len(self.val_data)))
        print("Timestamp: %s" % utils.get_time_hhmmss())

        print(self.string_fixer + " Params " + self.string_fixer)

        print("Learning Rate: %f" % self.params.lr)
        print("Dropout (p): %f" % self.params.dropout)
        print("Batch Size: %d" % self.params.batch_size)
        print("Epochs: %d" % self.params.epochs)
        print("Patience: %d" % self.params.patience)
        print("Resume: %s" % self.params.resume)

    def print_train_info(self, epoch, train_acc, train_loss, val_acc, val_loss):
        print((self.string_fixer + " Epoch: {0}/{1} " + self.string_fixer)
              .format(epoch + 1, self.params.epochs))
        print("Train Loss: %.8f, Train Acc: %.2f" % (train_loss, train_acc))
        print("Validation Loss: %.8f, Validation Acc: %.2f" % (val_loss, val_acc))
        self.early_stopping.print_info()
        print("Elapsed Time: %s" % (utils.get_time_hhmmss(self.start_time)))
        print("Current timestamp: %s" % (utils.get_time_hhmmss()))
