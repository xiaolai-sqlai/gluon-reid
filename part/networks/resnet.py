from __future__ import absolute_import

from mxnet.gluon import nn, HybridBlock
from mxnet import init, nd
from mxnet.gluon.model_zoo import vision


class ResNet(HybridBlock):
    __factory = {
        18: vision.resnet18_v1,
        34: vision.resnet34_v1,
        50: vision.resnet50_v1,
        101: vision.resnet101_v1,
        152: vision.resnet152_v1,
    }

    def __init__(self, depth, ctx, pretrained=True, num_features=0, num_classes=0, num_parts=1):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.num_parts = num_parts

        with self.name_scope():
            model = ResNet.__factory[depth](pretrained=pretrained, ctx=ctx).features[:-1]
            model[-1][0].body[0]._kwargs['stride'] = (1, 1)
            model[-1][0].downsample[0]._kwargs['stride'] = (1, 1)
            self.base = nn.HybridSequential()
            for m in model:
                self.base.add(m)

            #local
            self.feat = nn.HybridSequential()
            self.classify = nn.HybridSequential()
            for _ in range(num_parts):
                tmp = nn.HybridSequential()
                tmp.add(nn.GlobalMaxPool2D())
                feat = nn.Conv2D(channels=num_features, kernel_size=1, use_bias=False)
                feat.initialize(init.MSRAPrelu('in', 0), ctx=ctx)
                tmp.add(feat)
                bn = nn.BatchNorm()
                bn.initialize(init=init.Zero(), ctx=ctx)
                tmp.add(bn)
                tmp.add(nn.Flatten())
                self.feat.add(tmp)

                classifier = nn.Dense(num_classes, use_bias=False)
                classifier.initialize(init=init.Normal(0.001), ctx=ctx)
                self.classify.add(classifier)

            #global
            self.g_feat = nn.HybridSequential()
            self.g_classify = nn.HybridSequential()
            for _ in range(1):
                tmp = nn.HybridSequential()
                tmp.add(nn.GlobalAvgPool2D())
                feat = nn.Conv2D(channels=num_features, kernel_size=1, use_bias=False)
                feat.initialize(init.MSRAPrelu('in', 0), ctx=ctx)
                tmp.add(feat)
                bn = nn.BatchNorm(center=False, scale=False)
                bn.initialize(init=init.Zero(), ctx=ctx)
                tmp.add(bn)
                tmp.add(nn.Flatten())
                self.g_feat.add(tmp)

                classifier = nn.Dense(num_classes, use_bias=False)
                classifier.initialize(init=init.Normal(0.001), ctx=ctx)
                self.g_classify.add(classifier)


    def hybrid_forward(self, F, x):
        x = self.base(x)

        # local
        parts = nd.split(x, axis=2, num_outputs=self.num_parts)
        outputs = []
        for i in range(self.num_parts):
            part = self.feat[i](parts[i])
            if self.num_classes > 0:
                part = self.classify[i](part)
            outputs.append(part)

        # global
        g_part = self.g_feat[0](x)
        if self.num_classes > 0:
            g_part = self.g_classify[0](g_part)
        outputs.append(g_part)

        return outputs


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)