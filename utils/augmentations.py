import torch
from torchvision import transforms
import numpy as np


def compose_transformations(cfg, no_augmentations: bool):
    if no_augmentations:
        return transforms.Compose([Numpy2Torch()])

    transformations = []

    # cropping
    if cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE == 'none':
        transformations.append(UniformCrop(cfg.AUGMENTATION.CROP_SIZE))
    else:
        transformations.append(ImportanceRandomCrop(cfg.AUGMENTATION.CROP_SIZE))

    if cfg.AUGMENTATION.RANDOM_FLIP:
        transformations.append(RandomFlip())

    if cfg.AUGMENTATION.RANDOM_ROTATE:
        transformations.append(RandomRotate())

    if cfg.AUGMENTATION.COLOR_SHIFT:
        transformations.append(ColorShift())

    if cfg.AUGMENTATION.GAMMA_CORRECTION:
        transformations.append(GammaCorrection())

    transformations.append(Numpy2Torch())

    return transforms.Compose(transformations)


class Numpy2Torch(object):
    def __call__(self, args):
        images, labels = args
        images_tensor = torch.Tensor(images).permute(0, 3, 1, 2)
        labels_tensor = torch.Tensor(labels).permute(0, 3, 1, 2)
        return images_tensor, labels_tensor


class RandomFlip(object):
    def __call__(self, args):
        images, labels = args
        horizontal_flip = np.random.choice([True, False])
        vertical_flip = np.random.choice([True, False])

        if horizontal_flip:
            images = np.flip(images, axis=2)
            labels = np.flip(labels, axis=2)

        if vertical_flip:
            images = np.flip(images, axis=1)
            labels = np.flip(labels, axis=1)

        images = images.copy()
        labels = labels.copy()

        return images, labels


class RandomRotate(object):
    def __call__(self, args):
        images, labels = args
        k = np.random.randint(1, 4)  # number of 90 degree rotations
        images = np.rot90(images, k, axes=(1, 2)).copy()
        labels = np.rot90(labels, k, axes=(1, 2)).copy()
        return images, labels


class ColorShift(object):
    def __init__(self, min_factor: float = 0.5, max_factor: float = 1.5):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, args):
        img, label = args
        factors = np.random.uniform(self.min_factor, self.max_factor, img.shape[-1])
        img_rescaled = np.clip(img * factors[np.newaxis, np.newaxis, :], 0, 1).astype(np.float32)
        return img_rescaled, label


class GammaCorrection(object):
    def __init__(self, gain: float = 1, min_gamma: float = 0.25, max_gamma: float = 2):
        self.gain = gain
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, args):
        images, labels = args
        gamma = np.random.uniform(self.min_gamma, self.max_gamma, images.shape[-1])
        images = np.clip(np.power(images, gamma[np.newaxis, np.newaxis, np.newaxis, :]), 0, 1).astype(np.float32)
        return images, labels


# Performs uniform cropping on images
class UniformCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def random_crop(self, args):
        images, labels = args
        _, height, width, _ = labels.shape
        crop_limit_x = width - self.crop_size
        crop_limit_y = height - self.crop_size
        x = np.random.randint(0, crop_limit_x)
        y = np.random.randint(0, crop_limit_y)

        images_crop = images[:, y:y+self.crop_size, x:x+self.crop_size]
        labels_crop = labels[:, y:y+self.crop_size, x:x+self.crop_size]
        return images_crop, labels_crop

    def __call__(self, args):
        images, labels = self.random_crop(args)
        return images, labels


class ImportanceRandomCrop(UniformCrop):
    def __call__(self, args):

        sample_size = 20
        balancing_factor = 5

        random_crops = [self.random_crop(args) for _ in range(sample_size)]
        crop_weights = np.array([crop_label.sum() for _, crop_label in random_crops]) + balancing_factor
        crop_weights = crop_weights / crop_weights.sum()

        sample_idx = np.random.choice(sample_size, p=crop_weights)
        img, label = random_crops[sample_idx]

        return img, label
