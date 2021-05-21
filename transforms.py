import numpy as np
from skimage import color, exposure, transform
from constants import IMG_SIZE
from PIL import Image
from torch.legacy.nn import SpatialContrastiveNormalization
import torch


def preprocess_img(data):
    img = np.array(data)
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant')

    # roll color axis to axis 0
    # img = np.rollaxis(img, -1)
    return Image.fromarray(np.uint8(img * 255))


def gaussian_blur(kernel_width):
    # Initial kernel as provided
    initial_kernel = np.array([1, 2, 1], dtype=np.float32)
    final_kernel = initial_kernel

    # Convolve the kernel until we get the required kernel width
    # Each time we convolve the guassian kernel of width x with the
    # gaussian kernel given initially, we get a new guassian kernel
    # of width x + 2
    while final_kernel.size < kernel_width:
        final_kernel = np.convolve(final_kernel, initial_kernel)

    # Average the kernel
    final_kernel = (1.0 / np.sum(final_kernel)) * final_kernel

    # Reshape to 2d array
    final_kernel = final_kernel.reshape(-1, 1)

    return final_kernel


def normalize_local(img):
    norm_kernel = torch.from_numpy(gaussian_blur(7))
    norm = SpatialContrastiveNormalization(3, norm_kernel)
    batch_size = 200
    img = img.view(1, 3, 48, 48)
    img = norm.forward(img)
    img = img.view(3, 48, 48)
    return img


def data_test(img_path):
    img = Image.open(img_path)
    cv2.imshow('image', np.array(preprocess_img(img)))
    cv2.waitKey()
