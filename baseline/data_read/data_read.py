from __future__ import absolute_import
import os

from mxnet.gluon.data import dataset
from mxnet import image
from mxnet import nd


class ImageTxtDataset(dataset.Dataset):
    def __init__(self, root, items, flag=1, transform=None):
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._transform = transform
        self.items = items

    def __getitem__(self, idx):
        fpath = os.path.join(self._root, self.items[idx][0])
        img = image.imread(fpath, self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            img = self._transform(img)
        return img, label

    def __len__(self):
        return len(self.items)
