# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/24 22:29
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import torch.utils.data as data

from PIL import Image

import os
import os.path


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = []
    for e in os.listdir(dir):
        name = e.split('.')[0]

        if name not in classes:
            classes.append(name)

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    files = os.listdir(dir)

    for e in sorted(files):
        item = (e, class_to_idx[e.split('.')[0]])
        images.append(item)
        # labels.append(e.split('.')[0])

    return images


def _make_dataset(dir):
    images = []

    files = os.listdir(dir)

    for e in sorted(files):
        images.append(e)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x.xxx.ext
        root/class_x.xxy.ext
        root/class_x.xxz.ext

        root/class_y.123.ext
        root/class_y.nsdf3.ext
        root/class_y.asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, is_Train=True, transform=None, target_transform=None):
        self.train = is_Train

        if self.train:
            classes, class_to_idx = find_classes(root)
            # print('root', root)
            # print('classes:', classes)
            # print('class_to_idx:', class_to_idx)
            samples = make_dataset(root, class_to_idx)
            if len(samples) == 0:
                raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                                "Supported extensions are: " + ",".join(
                    extensions)))

            self.classes = classes
            self.class_to_idx = class_to_idx
        else:
            samples = _make_dataset(root)

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.train:
            path, target = self.samples[index]
            path = os.path.join(self.root, path)
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target
        else:
            path = self.samples[index]
            path = os.path.join(self.root, path)
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)

            return sample

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog.xxx.png
        root/dog.xxy.png
        root/dog.xxz.png

        root/cat.123.png
        root/cat.nsdf3.png
        root/cat.asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=pil_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
