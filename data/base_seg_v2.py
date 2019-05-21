"""Base segmentation dataset"""
import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter

from data.base import VisionDataset


class SegmentationDataset(VisionDataset):
    """Segmentation Base Dataset"""

    # pylint: disable=abstract-method
    def __init__(self, root, split, mode, transform, height, width):
        super(SegmentationDataset, self).__init__(root)
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.h = height
        self.w = width

    def _val_sync_transform(self, img, mask):
        img = img.resize((self.w, self.h), Image.BILINEAR)
        mask = mask.resize((self.w, self.h), Image.NEAREST)
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img = img.resize((self.w, self.h), Image.BILINEAR)
        mask = mask.resize((self.w, self.h), Image.NEAREST)
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        # return torch.from_numpy(np.array(img))
        return np.array(img)

    def _mask_transform(self, mask):
        # return torch.from_numpy(np.array(mask).astype('int32'))
        return np.array(mask).astype('int64')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


def ms_batchify_fn(data):
    """Multi-size batchify function"""
    if isinstance(data[0], (str, torch.Tensor)):
        return list(data)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [ms_batchify_fn(i) for i in data]
    raise RuntimeError('unknown datatype')
