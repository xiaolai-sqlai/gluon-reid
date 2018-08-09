# -*- coding: utf-8 -*-
from __future__ import print_function, division

import mxnet as mx
import numpy as np
from mxnet import gluon, image, nd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from networks import resnet18
from data_read import ImageTxtDataset

import time, os, sys
import scipy.io as sio
from sklearn import preprocessing

def get_data(batch_size, test_set, query_set):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_test = transforms.Compose([
        transforms.Resize(size=(128, 256), interpolation=1),
        transforms.ToTensor(),
        normalizer])

    test_imgs = ImageTxtDataset(data_dir+'test', test_set, transform=transform_test)
    query_imgs = ImageTxtDataset(data_dir+'query', query_set, transform=transform_test)

    test_data = gluon.data.DataLoader(test_imgs, batch_size, shuffle=False, last_batch='keep', num_workers=4)
    query_data = gluon.data.DataLoader(query_imgs, batch_size, shuffle=False, last_batch='keep', num_workers=4)
    return test_data, query_data


def load_network(network, ctx):
    network.load_params('params/resnet18.params', ctx=ctx, allow_missing=True, ignore_extra=True)
    return network


def fliplr(img):
    '''flip horizontal'''
    img_flip = nd.flip(img, axis=3)
    return img_flip


def extract_feature(model, dataloaders, ctx):
    count = 0
    features = []
    for img, _ in dataloaders:
        n = img.shape[0]
        count += n
        print(count)
        ff = np.zeros((n, 256))
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            f = model(img.as_in_context(ctx)).as_in_context(mx.cpu()).asnumpy()
            ff = ff+f
        features.append(ff)
    features = np.concatenate(features)
    return features/np.linalg.norm(features, axis=1, keepdims=True)


def get_id(img_path):
    cameras = []
    labels = []
    for path in img_path:
        cameras.append(int(path[0].split('_')[1][1]))
        labels.append(path[1])
    return cameras, labels


if __name__ == '__main__':
    batch_size = 256
    data_dir = '../../dataset/market1501/'
    gpu_ids = [0]

    # set gpu ids
    if len(gpu_ids)>0:
        context = mx.gpu()

    test_set = [(line, int(line.split('_')[0])) for line in os.listdir(data_dir+'test')]
    query_set = [(line, int(line.split('_')[0])) for line in os.listdir(data_dir+'query')]
    
    test_cam, test_label = get_id(test_set)
    query_cam, query_label = get_id(query_set)

    ######################################################################
    # Load Collected data Trained model
    model_structure = resnet18(ctx=context, pretrained=False, num_features=256)
    model = load_network(model_structure, context)

    # Extract feature
    test_loader, query_loader = get_data(batch_size, test_set, query_set)
    print('start test')
    test_feature = extract_feature(model, test_loader, context)
    print('start query')
    query_feature = extract_feature(model, query_loader, context)

    # Save to Matlab for check
    sio.savemat('result/test.mat', {'data':test_feature})
    sio.savemat('result/testID.mat', {'data':test_label})
    sio.savemat('result/testCam.mat', {'data':test_cam})

    sio.savemat('result/query.mat', {'data':query_feature})
    sio.savemat('result/queryID.mat', {'data':query_label})
    sio.savemat('result/queryCam.mat', {'data':query_cam})

