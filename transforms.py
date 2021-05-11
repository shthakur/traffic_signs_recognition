import numpy as np
from skimage import color, exposure, transform
from constants import IMG_SIZE
from PIL import Image


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


def data_test(img_path):
    img = Image.open(img_path)
    cv2.imshow('image', np.array(preprocess_img(img)))
    cv2.waitKey()
