from __future__ import absolute_import
import os, numbers

from mxnet.gluon.data import dataset
from mxnet import image, nd
from mxnet.gluon import Block


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


class Pad(Block):
    def __init__(self, padding):
        super(Pad, self).__init__()
        self.padding = padding

    def forward(self, x):
        if not isinstance(self.padding, (numbers.Number, tuple)):
            raise TypeError('Got inappropriate padding arg')
        shape = x.shape
        if isinstance(self.padding, numbers.Number):
            res = nd.zeros((shape[0]+2*self.padding, shape[1]+2*self.padding, shape[2]))
            res[self.padding:shape[0]+self.padding,self.padding:shape[1]+self.padding, :] = x
        if isinstance(self.padding, tuple):
            res = nd.zeros((shape[0]+2*self.padding[0], shape[1]+2*self.padding[1], shape[2]))
            res[self.padding:shape[0]+self.padding[0], self.padding:shape[1]+self.padding[1], :] = x
        return res


class RandomCrop(Block):
    def __init__(self, size):
        super(RandomCrop, self).__init__()
        self.size = size

    def forward(self, x):
        if not isinstance(self.size, (numbers.Number, tuple)):
            raise TypeError('Got inappropriate size arg')
        if isinstance(self.size, numbers.Number):
            size = (self.size, self.size)
        else:
            size = self.size
        return image.random_crop(x, size)[0]

