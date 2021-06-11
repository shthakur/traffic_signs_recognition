from __future__ import print_function
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from trainer import Trainer
from modules.utils import utils
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch GTSRB evaluation script')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--resume', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form X.pth")
parser.add_argument('--outfile', type=str, default='gtsrb_kaggle.csv', metavar='D',
                    help="name of the output csv file")


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def main():
    cuda = torch.cuda.is_available()
    params = parser.parse_args()

    # For sake of trainer loading
    params.cnn = ''
    params.st = False
    params.lr = 0.001
    params.patience = 10

    test_dir = params.data + '/test_images'
    trainer = Trainer(params)
    trainer.load()

    output_file = open(params.outfile, "w")
    output_file.write("Filename,ClassId\n")
    trainer.model.eval()

    for f in tqdm(os.listdir(test_dir)):
        if 'ppm' in f:
            data = utils.val_data_transforms(pil_loader(test_dir + '/' + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            data = data.permute(0, 3, 2, 1)
            data, _ = utils.preprocess_dataset(data.numpy(), use_tqdm=False)
            data = torch.from_numpy(data).permute(0, 3, 2, 1)
            data = Variable(data, volatile=True)
            if cuda:
                data = data.cuda()
            output = trainer.model(data)
            pred = output.data.max(1, keepdim=True)[1]

            file_id = f[0:5]
            output_file.write("%s,%d\n" % (file_id, pred[0][0]))
    trainer.model.train()
    output_file.close()


if __name__ == '__main__':
    main()
