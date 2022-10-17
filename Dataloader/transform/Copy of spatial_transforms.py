import random
import math
import numbers
import collections
import numpy as np
import torch
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None


import random

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            t.randomize_parameters()
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()

class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a integer Tensor (T x H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (T x C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, clip):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """

        clip = clip.type(torch.FloatTensor).div(self.norm_value)
        print(clip.shape)
        return clip.permute(0, 3, 1, 2).contiguous()

    def randomize_parameters(self):
        pass

class ReturnVideo(object):

    def __init__(self):
        pass

    def __call__(self, tup):
        return tup[0]

    def randomize_parameters(self):
        pass

class VideoRandomCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        T,C,H,W = clip.shape

        x1 = int(self.tl_x * (H - self.size))
        y1 = int(self.tl_y * (W - self.size))
        x2 = x1 + self.size
        y2 = y1 + self.size

        clip = clip[:,:,x1:x2,y1:y2]

        return clip

    def randomize_parameters(self):
        self.tl_x = random.random()
        self.tl_y = random.random()


class RandomHorizontalFlip(object):

    def __call__(self, clip):
        if self.p < 0.5:
            return clip.flip([-1])
        return clip

    def randomize_parameters(self):
        self.p = random.random()