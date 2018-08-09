from __future__ import division

import argparse, datetime, random
import logging
logging.basicConfig(level=logging.INFO)

import mxnet as mx
from mxnet import gluon, image, nd
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.data.vision import transforms
from mxnet import autograd


from networks import resnet50, TripletLoss
from data_read import ImageTxtDataset, RandomIdentitySampler

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--train-data', type=str, default='txt/train.txt',
                    help='training record file to use, required for imagenet.')
parser.add_argument('--val-data', type=str, default='txt/val.txt',
                    help='validation record file to use, required for imagenet.')
parser.add_argument('--img-height', type=int, default=384,
                    help='the height of image for input')
parser.add_argument('--img-width', type=int, default=128,
                    help='the width of image for input')
parser.add_argument('--batch-size', type=int, default=16,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-workers', type=int, default=36,
                    help='the number of workers for data loader')
parser.add_argument('--num-gpus', type=int, default=4,
                    help='number of gpus to use.')
parser.add_argument('--triplet', type=bool, default=True,
                    help='number of gpus to use.')
parser.add_argument('--epochs', type=str, default="20,110,140,160")
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate. default is 1e-1.')
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
parser.add_argument('--hybridize', type=bool, default=False)

def get_data_iters(batch_size):
    train_set = [(name.split()[0], int(name.split()[1])) for name in open(opt.train_data).readlines()]
    val_set = [(name.split()[0], int(name.split()[1])) for name in open(opt.val_data).readlines()]

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.Resize(size=(opt.img_width, opt.img_height), interpolation=1),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        normalizer])

    transform_test = transforms.Compose([
        transforms.Resize(size=(opt.img_width, opt.img_height), interpolation=1),
        transforms.ToTensor(),
        normalizer])

    train_imgs = ImageTxtDataset(opt.data_dir, train_set, transform=transform_train)
    val_imgs = ImageTxtDataset(opt.data_dir, val_set, transform=transform_test)

    train_data = gluon.data.DataLoader(train_imgs, batch_size, sampler=RandomIdentitySampler(train_set, 4), last_batch='discard', num_workers=opt.num_workers)
    val_data = gluon.data.DataLoader(val_imgs, batch_size, sampler=RandomIdentitySampler(val_imgs, 4), last_batch='discard', num_workers=opt.num_workers)

    return train_data, val_data


def validate(val_data, net, criterion1, criterion2, ctx):
    val_loss = 0.0
    for data, label in val_data:
        data_list = gluon.utils.split_and_load(data, ctx)
        label_list = gluon.utils.split_and_load(label, ctx)

        losses = []
        accurays = []
        for i in range(opt.num_gpus):
            outputs, features = net(data_list[i])
            temp_loss = []
            num = len(outputs)
            for j in range(8):
                temp_loss.append(criterion1[j](outputs[j], label_list[i]))
            if opt.triplet:
                num += len(features)
                for j in range(3):
                    temp_loss.append(criterion2[j](features[j], label_list[i]))
            loss = sum(temp_loss) / num
            losses.append(loss)
            temp_acc = sum([nd.mean(X.argmax(axis=1)==label_list[i].astype('float32')).asscalar() for X in outputs])/8
            accurays.append(temp_acc)

        loss_list = [l.mean().asscalar() for l in losses]
        val_loss += sum(loss_list) / len(loss_list)

    return val_loss/len(val_data), sum(accurays)/len(accurays)


def main(net, batch_size, epochs, opt, ctx):
    train_data, val_data = get_data_iters(batch_size)
    if opt.hybridize:
        net.hybridize()

    #trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': opt.lr, 'wd': opt.wd})
    criterion1 = []
    for _ in range(8):
        criterion1.append(gluon.loss.SoftmaxCrossEntropyLoss())
    criterion2 = []
    if opt.triplet:
        for _ in range(3):
            criterion2.append(TripletLoss())

    lr = opt.lr
    minlr = lr*0.01
    dlr = (lr-minlr)/(epochs[0]-1)

    prev_time = datetime.datetime.now()
    for epoch in range(epochs[-1]):
        _loss = 0.
        if epoch<epochs[0]:
            lr = minlr + dlr*epoch
        else:
            if epoch in epochs[1:]:
                lr = lr * opt.lr_decay
        trainer.set_learning_rate(lr)

        for data, label in train_data:
            data_list = gluon.utils.split_and_load(data, ctx)
            label_list = gluon.utils.split_and_load(label, ctx)
            with autograd.record():
                losses = []
                for i in range(opt.num_gpus):
                    outputs, features = net(data_list[i])
                    temp_loss = []
                    num = len(outputs)
                    for j in range(len(outputs)):
                        temp_loss.append(criterion1[j](outputs[j], label_list[i]))
                    if opt.triplet:
                        num += len(features)
                        for j in range(len(features)):
                            temp_loss.append(criterion2[j](features[j], label_list[i]))
                    loss = sum(temp_loss) / num
                    losses.append(loss)

            for l in losses:
                l.backward()
            trainer.step(batch_size)
            _loss_list = [l.mean().asscalar() for l in losses]
            _loss += sum(_loss_list) / len(_loss_list)

        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        __loss = _loss/len(train_data)

        if val_data is not None:
            val_loss, val_accuray = validate(val_data, net, criterion1, criterion2, ctx)
            epoch_str = ("Epoch %d. Train loss: %f, Val loss %f, Val accuray %f, " % (epoch, __loss , val_loss, val_accuray))
        else:
            epoch_str = ("Epoch %d. Train loss: %f, " % (epoch, __loss))

        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))

    net.save_parameters("params/resnet50.params")


if __name__ == '__main__':
    opt = parser.parse_args()
    logging.info(opt)
    random.seed(opt.seed)
    mx.random.seed(opt.seed)

    batch_size = opt.batch_size
    num_gpus = opt.num_gpus
    context = [mx.gpu(i) for i in range(num_gpus)]
    epochs = [int(i) for i in opt.epochs.split(',')]
    batch_size *= max(1, num_gpus)

    net = resnet50(ctx=context, num_features=256, num_classes=751)
    main(net, batch_size, epochs, opt, context)
