from __future__ import print_function
import argparse
import os
import numpy as np

from modules.utils import utils
from modules.Extender import Extender
from constants import NUM_CLASSES


parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data/train_images', metavar='D',
                    help="folder where data is train data is located")
parser.add_argument('--save_loc', type=str, default='.', metavar='P',
                    help="folder where all the pickles will be stored")
parser.add_argument('--intensity', type=float, default=0.75, metavar='LR',
                    help='Augmentation intesity (default: 0.75)')
parser.add_argument('--class_count', type=int, default=10000, metavar='LR',
                    help='Each class will have this number of data after extension and balancing (Default: 10k)')


def main():
    args = parser.parse_args()
    train_pickle_path = os.path.join(args.save_loc, "train.p")
    # utils.pickle_data_from_folder(args.data, train_pickle_path)
    print("Saved train.p from original data folder")
    print("Now extending data")
    x, y = utils.load_pickled_data(train_pickle_path, ["features", "labels"])
    extender = Extender(x, y, 1, args.intensity)
    x_extended, y_extended = extender.flip()
    print("Data extension complete")
    utils.pickle_data(x_extended, y_extended,
                      os.path.join(args.save_loc, "train_extended.p"))

    # Generate "class_count (default: 10k)" augmented plus original images for each class
    x_balanced, y_balanced = extender.extend_and_balance(
        custom_counts=np.array([args.class_count] * NUM_CLASSES))
    utils.pickle_data(x_balanced, y_balanced,
                      os.path.join(args.save_loc, "train_balanced_" +
                                   str(args.intensity) + ".p"))

    x_preprocessed, y_preprocessed = utils.preprocess_dataset(x_balanced,
                                                              y_balanced)
    utils.pickle_data(x_preprocessed, y_preprocessed,
                      os.path.join(args.save_loc,
                                   "train_balanced_preprocessed_" +
                                   str(args.intensity) + ".p"))


if __name__ == '__main__':
    main()
