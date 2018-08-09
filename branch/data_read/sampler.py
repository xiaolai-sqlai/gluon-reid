#/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from collections import defaultdict

import numpy as np
from mxnet.gluon.data import Sampler


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(data_source):
            self.index_dic[pid].append(index) # 把对应的索引插入index_dic中的每个pid
        self.pids = list(self.index_dic.keys()) # 得到所有的pid的名字
        self.num_samples = len(self.pids) # 得到pid的数目

    def __iter__(self):
        indices = range(0, self.num_samples)
        np.random.shuffle(indices) # 打乱排序的索引
        ret = []
        for i in indices:
            pid = self.pids[i] # 得到实际的pid号
            t = self.index_dic[pid] # 得到实际的pid中对应的图片
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            elif len(t) > 0:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_samples * self.num_instances
