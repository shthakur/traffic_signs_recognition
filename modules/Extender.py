import random
import numpy as np

from skimage.transform import rotate, warp, ProjectiveTransform


class Extender:
    def __init__(self, data_images, data_labels, ratio=0.5, intensity=0.75):
        self.x = data_images
        self.y = data_labels
        self.intensity = intensity
        self.ratio = ratio

        # These classes can be horizontally flipped for new images
        # Contains yield sign, ice, traffic signals etc
        self.horizontally_flippable_classes = [11, 12, 13, 15, 17, 18,
                                               22, 26, 30, 35]

        # These ones can vertically flipped for new images
        # Contains 30 limit, 80 limit, no vehicles
        self.vertically_flippable_classes = np.array([1, 5, 12, 15, 17])

        # These ones need to be flipped both vertically and horizontally
        # to get same image
        self.both_flippable = [32, 40]

        # These ones contains pair which first can be generated by
        # horizontally flipping the other. For e.g turn left and
        # turn right
        self.flip_exchangeable = np.array([
            (19, 20),
            (20, 19),
            (33, 34),
            (34, 33),
            (36, 37),
            (37, 36),
            (38, 39),
            (39, 38)
        ])

    def extend_and_balance(self, custom_counts=None):
        print("Extending and balancing dataset with intesity", self.intensity)
        x, y = self.flip()
        _, class_counts = np.unique(y, return_counts=True)
        max_count = max(class_counts)

        if custom_counts is None:
            total = max_count * NUM_CLASSES
        else:
            total = np.sum(custom_counts)

        x_balanced = np.empty([0, x.shape[1], x.shape[2], x.shape[3]],
                              dtype=np.float32)
        y_balanced = np.empty([0], dtype=y.dtype)

        for c, class_count in zip(range(NUM_CLASSES), tqdm(class_counts)):
            x_org = (x[y == c] / 255.).astype(np.float32)
            y_org = y[y == c]

            x_balanced = np.append(x_balanced, x_org, axis=0)

            max_count = max_count if custom_counts is None else custom_counts[c]
            for i in range(max_count // class_count):
                x_mod = self.rotate(x_org)
                x_mod = self.projection_transform(x_mod)
                x_balanced = np.append(x_balanced, x_mod, axis=0)

            if max_count % class_count > 0:
                x_mod = self.rotate(x_org[:(max_count % class_count)])
                x_mod = self.projection_transform(x_mod)

                x_balanced = np.append(x_balanced, x_mod, axis=0)

            extension = np.full(x_balanced.shape[0] - y_balanced.shape[0],
                                c, dtype=y_balanced.dtype)
            y_balanced = np.append(y_balanced, extension)

            del x_org
            del y_org

        return (x_balanced * 255).astype(np.uint8), y_balanced

    def flip(self):
        x = np.empty([0, self.x.shape[1], self.x.shape[2],
                      self.x.shape[3]],
                     dtype=self.x.dtype)
        y = np.empty([0], dtype=self.y.dtype)

        for c in range(NUM_CLASSES):
            # Add existing data
            x = np.append(x, self.x[self.y == c], axis=0)

            if c in self.horizontally_flippable_classes:
                # Flip columns and append
                x = np.append(x, self.x[self.y == c][:, :, ::-1, :],
                              axis=0)

            if c in self.vertically_flippable_classes:
                # Flip rows and append
                x = np.append(x, self.x[self.y == c][:, ::-1, :, :],
                              axis=0)

            if c in self.flip_exchangeable[:, 0]:
                flip_c = self.flip_exchangeable[self.flip_exchangeable[:, 0] == c]
                flip_c = flip_c[0][1]

                # Flip other class horizontally
                x = np.append(x, self.x[self.y == flip_c][:, :, ::-1, :],
                              axis=0)

            if c in self.both_flippable:
                # Flip both rows and columns
                x = np.append(x, self.x[self.y == c][:, ::-1, ::-1, :],
                              axis=0)

            # Extend y now
            y = np.append(y, np.full(x.shape[0] - y.shape[0], c,
                                     dtype=int))

        return (x, y)

    def rotate(self, x):
        indices = np.random.choice(x.shape[0], int(x.shape[0] * self.ratio),
                                   replace=False)

        # If we rotate more than 30 degrees, context is lost.
        change = 30. * self.intensity
        x_return = np.empty(x.shape, dtype=x.dtype)
        for i in indices:
            x_return[i] = rotate(x[i], random.uniform(-change, change),
                                 mode="edge")

        return x_return

    def projection_transform(self, x):
        image_size = x.shape[1]

        change = image_size * 0.3 * self.intensity

        x_return = np.empty(x.shape, dtype=x.dtype)

        indices = np.random.choice(x.shape[0], int(x.shape[0] * self.ratio),
                                   replace=False)
        for i in indices:
            changes = []
            for _ in range(8):
                changes.append(random.uniform(-change, change))

            transform = ProjectiveTransform()
            transform.estimate(np.array(
                (
                    (changes[0], changes[1]),  # top left
                    (changes[2], image_size - changes[3]),  # bottom left
                    (image_size - changes[4], changes[5]),  # top right
                    (image_size - changes[6], image_size - changes[7])  # bottom right
                )), np.array(
                (
                    (0, 0),
                    (0, image_size),
                    (image_size, 0),
                    (image_size, image_size)
                ))
            )

            x_return[i] = warp(x[i], transform,
                               output_shape=(image_size, image_size),
                               order=1, mode="edge")

        return x_return
