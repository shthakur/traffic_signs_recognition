from __future__ import print_function
import argparse
from modules.utils import utils
from modules.extender import Extender


parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data/train_images', metavar='D',
                    help="folder where data is train data is located")
parser.add_argument('--save_loc', type=str, default='.', metavar='P',
                    help="folder where all the pickles will be stored")
parser.add_argument('--intesity', type=float, default=0.75, metavar='LR',
                    help='Augmentation intesity (default: 0.75)')


def main():
    args = parser.parse_args()
    train_pickle_path = os.path.join(args.save_loc, "train.p")
    utils.pickle_data_from_folder(args.data, train_pickle_path)
    x, y = utils.load_pickled_data(train_pickle_path, ["features", "labels"])
    extender = Extender(x, y, 1, args.intensity)
    x_extended, y_extended = extender.flip()
    utils.pickle_data(x_extended, y_extended,
                      os.path.join(args.save_loc, "train_extended.p"))

    # Generate 10k augmented plus original images for each class
    x_balanced, y_balanced = extender.extend_and_balance(
        custom_counts=np.array([10000] * NUM_CLASSES))
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
