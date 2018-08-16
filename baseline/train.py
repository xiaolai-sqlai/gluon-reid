from __future__ import division

import argparse, datetime, os
import logging
logging.basicConfig(level=logging.INFO)

import mxnet as mx
import numpy
from mxnet import gluon, image, nd
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.data.vision import transforms
from mxnet import autograd


from networks import resnet18, resnet34, resnet50
from data_read import ImageTxtDataset, Pad, RandomCrop

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--train-data', type=str, default='txt/train.txt',
                    help='training record file to use, required for imagenet.')
parser.add_argument('--val-data', type=str, default='txt/val.txt',
                    help='validation record file to use, required for imagenet.')
parser.add_argument('--img-height', type=int, default=256,
                    help='the height of image for input')
parser.add_argument('--img-width', type=int, default=128,
                    help='the width of image for input')
parser.add_argument('--batch-size', type=int, default=8,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-workers', type=int, default=32,
                    help='the number of workers for data loader')
parser.add_argument('--num-gpus', type=int, default=4,
                    help='number of gpus to use.')
parser.add_argument('--warmup', type=bool, default=True,
                    help='number of training epochs.')
parser.add_argument('--epochs', type=str, default="10,30,55,80")
parser.add_argument('--lr', type=float, default=3.5e-4,
                    help='learning rate. default is 0.1.')
parser.add_argument('-momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay rate. default is 5e-4.')
parser.add_argument('--seed', type=int, default=613,
                    help='random seed to use. Default=613.')
parser.add_argument('--kvstore', type=str, default='device',
                    help='kvstore to use for trainer/module.')
parser.add_argument('--data-dir', default='../dataset/market1501/train', type=str, metavar='PATH')
parser.add_argument('--lr-decay', type=int, default=0.1)
parser.add_argument('--hybridize', type=bool, default=True)


def concat_and_load(data, use_cpu=True, gpu_id=0):
    """Splits an NDArray into `len(ctx_list)` slices along `batch_axis` and loads
    each slice to one context in `ctx_list`.
    Parameters
    ----------
    data : NDArray
        A batch of data in different gpu.
    use_cpu : bool, default True
        Use cpu or gpu.
    gpu_id : int, default 0
        Which gpu to use.
    Returns
    -------
    NDArray
        All data in one device.
    """
    if isinstance(data, list):
        if use_cpu:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(gpu_id)
        data = [x.as_in_context(ctx) for x in data]
        return nd.concatenate(data)


def get_data_iters(batch_size):
    train_set = [(name.split()[0], int(name.split()[1])) for name in open(opt.train_data).readlines()]
    val_set = [(name.split()[0], int(name.split()[1])) for name in open(opt.val_data).readlines()]

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.Resize(size=(opt.img_width, opt.img_height), interpolation=1),
        transforms.RandomFlipLeftRight(),
        Pad(10),
        RandomCrop(size=(opt.img_width, opt.img_height)),
        transforms.ToTensor(),
        normalizer])

    transform_test = transforms.Compose([
        transforms.Resize(size=(opt.img_width, opt.img_height), interpolation=1),
        transforms.ToTensor(),
        normalizer])

    train_imgs = ImageTxtDataset(opt.data_dir, train_set, transform=transform_train)
    val_imgs = ImageTxtDataset(opt.data_dir, val_set, transform=transform_test)

    train_data = gluon.data.DataLoader(train_imgs, batch_size, shuffle=True, last_batch='discard', num_workers=opt.num_workers)
    val_data = gluon.data.DataLoader(val_imgs, batch_size, shuffle=True, last_batch='keep', num_workers=opt.num_workers)

    return train_data, val_data


def validate(val_data, net, criterion, ctx):
    _loss = 0.0
    for data, label in val_data:
        data_list = gluon.utils.split_and_load(data, ctx)
        label_list = gluon.utils.split_and_load(label, ctx)

        with autograd.predict_mode():
            outputs = concat_and_load([net(X) for X in data_list])
            labels = concat_and_load(label_list)
            loss = criterion(outputs, labels)
        accuray = nd.mean(outputs.argmax(axis=1)==labels.astype('float32')).asscalar()

        _loss += loss.mean().asscalar()

    return _loss/len(val_data), sum(accuray)/len(accuray)


def main(net, batch_size, epochs, opt, ctx):
    train_data, val_data = get_data_iters(batch_size)
    if opt.hybridize:
        net.hybridize()

    # trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': opt.lr, 'wd': opt.wd})

    criterion = gluon.loss.SoftmaxCrossEntropyLoss()

    lr = opt.lr
    if opt.warmup:
        minlr = lr*0.01
        dlr = (lr-minlr)/(epochs[0]-1)

    prev_time = datetime.datetime.now()
    for epoch in range(epochs[-1]):
        _loss = 0.
        if opt.warmup:
            if epoch<epochs[0]:
                lr = minlr + dlr*epoch
        if epoch in epochs[1:]:
            lr = lr * opt.lr_decay
        trainer.set_learning_rate(lr)

        for data, label in train_data:
            data_list = gluon.utils.split_and_load(data, ctx)
            label_list = gluon.utils.split_and_load(label, ctx)
            with autograd.record():
                outputs = concat_and_load([net(X) for X in data_list])
                labels = concat_and_load(label_list)
                loss = criterion(outputs, labels)
            
            loss.backward()
            trainer.step(batch_size)
            _loss += loss.mean().asscalar()
            # accuray = nd.mean(outputs.argmax(axis=1)==labels.astype('float32')).asscalar()
            # print(_loss, accuray)

        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        __loss = _loss/len(train_data)

        if val_data is not None:
            val_loss, val_accuray = validate(val_data, net, criterion, ctx)
            epoch_str = ("Epoch %d. Train loss: %f, Val loss %f, Val accuray %f, " % (epoch, __loss , val_loss, val_accuray))
        else:
            epoch_str = ("Epoch %d. Train loss: %f, " % (epoch, __loss))

        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))

    if not os.path.exists("params"):
        os.mkdir("params")
    net.save_parameters("params/resnet50.params")


if __name__ == '__main__':
    opt = parser.parse_args()
    logging.info(opt)
    mx.random.seed(opt.seed)

    batch_size = opt.batch_size
    num_gpus = opt.num_gpus
    epochs = [int(i) for i in opt.epochs.split(',')]
    batch_size *= max(1, num_gpus)

    context = [mx.gpu(i) for i in range(num_gpus)]
    net = resnet50(ctx=context, num_classes=751)
    main(net, batch_size, epochs, opt, context)
