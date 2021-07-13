from constants import IMG_SIZE, NUM_CLASSES
from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils as vutils, datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage import exposure
from PIL import Image
from data import TrafficSignsDataset
from constants import IMG_SIZE, NUM_CLASSES
from tqdm import tqdm

import random
import warnings
import os
import pickle
import time
import shutil


class Utils:
    def __init__(self):
        self.train_data_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])
        self.val_data_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

    def load_pickled_data(self, file, columns):
        with open(file, mode='rb') as f:
            dataset = pickle.load(f)
        return tuple(map(lambda c: dataset[c], columns))

    def get_dataset(self, params):
        if params.use_pickle:
            data_images, data_labels = self.load_pickled_data(
                params.train_pickle, ['features', 'labels'])
            train_images, val_images, train_labels, val_labels = train_test_split(data_images,
                                                                                  data_labels,
                                                                                  test_size=0.25)
            return TrafficSignsDataset(train_images, train_labels), TrafficSignsDataset(val_images, val_labels)
        else:
            train_dataset = datasets.ImageFolder(params.data + '/train_images',
                                                 transform=self.train_data_transforms)
            val_dataset = datasets.ImageFolder(params.data + '/val_images',
                                               transform=self.val_data_transforms)
            return train_dataset, val_dataset

    def pickle_data(self, x, y, save_loc):
        print("Saving pickle at " + save_loc)
        save = {"features": x, "labels": y}

        with open(save_loc, "wb") as f:
            pickle.dump(save, f)

    def pickle_data_from_folder(self, data_folder, save_loc):
        if not os.path.isdir(data_folder):
            print("Data folder must be a folder and should contains sub folders for each label")
            return

        resize_transform = transforms.Resize((IMG_SIZE, IMG_SIZE))
        sub_folders = os.listdir(data_folder)

        count = 0
        for sub_folder in sub_folders:
            sub_folder = os.path.join(data_folder, sub_folder)

            if not os.path.isdir(sub_folder):
                continue
            label = int(sub_folder.split("/")[-1])

            for image in os.listdir(sub_folder):
                count += 1

        save = {"features": np.empty([count, IMG_SIZE, IMG_SIZE, 3], dtype=np.uint8),
                "labels": np.empty([count], dtype=int)}
        i = 0
        for sub_folder in sub_folders:
            sub_folder = os.path.join(data_folder, sub_folder)

            if not os.path.isdir(sub_folder):
                continue
            label = int(sub_folder.split("/")[-1])
            for image in os.listdir(sub_folder):
                image = os.path.join(sub_folder, image)
                pic = Image.open(image)
                pic = resize_transform(pic)
                pic = np.array(pic)
                save["features"][i] = pic
                save["labels"][i] = label
                i += 1

        with open(save_loc, "wb") as f:
            pickle.dump(save, f)

    def get_dataset_from_file(self, file):
        data_images, data_labels = self.load_pickled_data(file, ['features', 'labels'])

        return TrafficSignsDataset(data_images, data_labels)

    def get_convnet_output_size(self, network, input_size=IMG_SIZE):
        input_size = input_size or IMG_SIZE

        if type(network) != list:
            network = [network]

        in_channels = network[0].conv.in_channels

        output = Variable(torch.ones(1, in_channels, input_size, input_size))
        output.require_grad = False
        for conv in network:
            output = conv.forward(output)

        return np.asscalar(np.prod(output.data.shape)), output.data.size()[2]

    def get_time_hhmmss(self, start=None):
        """
        Calculates time since `start` and formats as a string.
        """
        if start is None:
            return time.strftime("%Y/%m/%d %H:%M:%S")
        end = time.time()
        m, s = divmod(end - start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        epoch = state['epoch']
        print("=> Saving model to %s" % filename)

        if is_best:
            print("=> The model just saved has performed best on validation set" +
                  " till now")
            shutil.copyfile(filename, 'model_best.pth.tar')

        return filename

    def load_checkpoint(self, resume):
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))

            # Helps with loading models trained on GPU to run on CPU
            if not torch.cuda.is_available():
                checkpoint = torch.load(resume, map_location=lambda storage, location: storage)
            else:
                checkpoint = torch.load(resume)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
            return checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            return None

    def imshow(self, img):
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def plot_images(self, trainloader):
        # get some random training images
        dataiter = iter(trainloader)
        images, labels = dataiter.next()

        # show images
        imshow(vutils.make_grid(images))

    def plot_classes(self, loader):
        class_images = {}

        for images, labels in loader:
            for image, label in zip(images, labels):
                if len(class_images) == NUM_CLASSES:
                    break
                if label not in class_images:
                    class_images[label] = [image]
                else:
                    class_images[label].append(image)

            if len(class_images) == NUM_CLASSES:
                break

        final_images = [random.choice(class_images[i]) for i in sorted(class_images)]

        imshow(vutils.make_grid(torch.stack(final_images)))

    def preprocess_dataset(self, X, y=None, use_tqdm=True):
        # Convert to single channel Y
        X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]

        # Scale
        X = (X / 255.).astype(np.float32)

        # Don't want to use tqdm while generating csv
        if use_tqdm:
            preprocess_range = tqdm(range(X.shape[0]))
        else:
            preprocess_range = range(X.shape[0])

        # Ignore warnings, see http://scikit-image.org/docs/dev/user_guide/data_types.html
        for i in preprocess_range:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X[i] = exposure.equalize_adapthist(X[i])

        if y is not None:
            # Convert to one-hot encoding. Convert back with `y = y.nonzero()[1]`
            y = np.eye(NUM_CLASSES)[y]
            X, y = shuffle(X, y)

        # Add a single grayscale channel
        X = X.reshape(X.shape + (1,))
        return X, y


utils = Utils()
