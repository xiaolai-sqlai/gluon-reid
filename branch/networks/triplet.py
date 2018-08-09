from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from mxnet import nd
import numpy as np


def euclidean_dist(x, y):
    m, n = x.shape[0], y.shape[0]
    xx = nd.power(x, 2).sum(axis=1, keepdims=True).broadcast_to((m, n))
    yy = nd.power(y, 2).sum(axis=1, keepdims=True).broadcast_to((n, m)).T
    dist = xx + yy
    dist = dist - 2 * nd.dot(x, y.T)
    dist = dist.clip(a_min=1e-12, a_max=1e12).sqrt()
    return dist


def hard_example_mining(dist_mat, labels):
    assert len(dist_mat.shape) == 2
    assert dist_mat.shape[0] == dist_mat.shape[1]
    N = dist_mat.shape[0]

    # shape [N, N]
    is_pos = nd.equal(labels.broadcast_to((N, N)), labels.broadcast_to((N, N)).T).astype('float32')
    is_neg = nd.not_equal(labels.broadcast_to((N, N)), labels.broadcast_to((N, N)).T).astype('float32')

    dist_pos = dist_mat * is_pos
    dist_ap = nd.max(dist_pos, axis=1)

    dist_neg = dist_mat * is_neg + nd.max(dist_mat, axis=1, keepdims=True) * is_pos
    dist_an = nd.min(dist_neg, axis=1)

    return dist_ap, dist_an


class TripletLoss(object):
    def __init__(self, margin=1.2):
        self.margin = margin

    def __call__(self, global_feat, labels):
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        loss = nd.relu(dist_ap - dist_an + self.margin)
        return loss