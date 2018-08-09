from __future__ import absolute_import

from mxnet.gluon import nn, HybridBlock, Parameter
from mxnet import init, nd
from mxnet.gluon.model_zoo import vision



class NormLinear(HybridBlock):
    def __init__(self, num_features, num_classes, ctx, scale=15):
        super(NormLinear, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        with self.name_scope():
            self.weight = Parameter('norm_weight', shape=(num_classes, num_features))
            self.weight.initialize(init.Xavier(magnitude=2.24), ctx=ctx)

    def hybrid_forward(self, F, x, weight):
        feat = F.L2Normalization(x, mode='instance')
        norm_weight = F.L2Normalization(weight, mode='instance')
        cosine = F.FullyConnected(feat, norm_weight, no_bias=True, num_hidden=self.num_classes)
        output = self.scale * cosine
        return output # size=(Batch, Class)


class ResNet(HybridBlock):
    __factory = {
        18: vision.resnet18_v1,
        34: vision.resnet34_v1,
        50: vision.resnet50_v1,
        101: vision.resnet101_v1,
        152: vision.resnet152_v1,
    }

    def __init__(self, depth, ctx, pretrained=True, num_features=0, num_classes=0):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes

        with self.name_scope():
            model1 = ResNet.__factory[depth](pretrained=pretrained, ctx=ctx).features[:-1]
            model1[-1][0].body[0]._kwargs['stride'] = (1, 1)
            model1[-1][0].downsample[0]._kwargs['stride'] = (1, 1)

            model2 = ResNet.__factory[depth](pretrained=pretrained, ctx=ctx).features[:-1]
            model2[-1][0].body[0]._kwargs['stride'] = (1, 1)
            model2[-1][0].downsample[0]._kwargs['stride'] = (1, 1)

            model3 = ResNet.__factory[depth](pretrained=pretrained, ctx=ctx).features[:-1]
            model3[-1][0].body[0]._kwargs['stride'] = (1, 1)
            model3[-1][0].downsample[0]._kwargs['stride'] = (1, 1)

            #backbone
            self.base = nn.HybridSequential()
            for m in model1[:-2]:
                self.base.add(m)
            self.base.add(model1[-2][0])

            #branch 1
            self.branch1 = nn.HybridSequential()
            for m in model1[-2][1:]:
                self.branch1.add(m)
            for m in model1[-1]:
                self.branch1.add(m)

            #branch 2
            self.branch2 = nn.HybridSequential()
            for m in model2[-2][1:]:
                self.branch2.add(m)
            for m in model2[-1]:
                self.branch2.add(m)

            #branch 3
            self.branch3 = nn.HybridSequential()
            for m in model3[-2][1:]:
                self.branch3.add(m)
            for m in model3[-1]:
                self.branch3.add(m)
            

            #local
            self.feat = nn.HybridSequential()
            self.classify = nn.HybridSequential()
            for _ in range(5):
                tmp = nn.HybridSequential()
                tmp.add(nn.GlobalMaxPool2D())
                feat = nn.Conv2D(channels=num_features, kernel_size=1, use_bias=False)
                feat.initialize(init=init.MSRAPrelu('in', 0), ctx=ctx)
                tmp.add(feat)
                bn = nn.BatchNorm()
                bn.initialize(init=init.Zero(), ctx=ctx)
                tmp.add(bn)
                tmp.add(nn.Flatten())
                self.feat.add(tmp)

                classifier = nn.Dense(num_classes, use_bias=False)
                classifier.weight.initialize(init=init.Normal(0.001), ctx=ctx)
                self.classify.add(classifier)

            #global
            self.g_feat = nn.HybridSequential()
            self.g_classify = nn.HybridSequential()
            for _ in range(3):
                tmp = nn.HybridSequential()
                tmp.add(nn.GlobalAvgPool2D())
                feat = nn.Conv2D(channels=num_features, kernel_size=1, use_bias=False)
                feat.initialize(init=init.MSRAPrelu('in', 0), ctx=ctx)
                tmp.add(feat)
                bn = nn.BatchNorm(center=False, scale=False)
                bn.initialize(init=init.Zero(), ctx=ctx)
                tmp.add(bn)
                tmp.add(nn.Flatten())
                self.g_feat.add(tmp)

                classifier = nn.Dense(num_classes, use_bias=False)
                classifier.initialize(init=init.Normal(0.001), ctx=ctx)
                self.g_classify.add(classifier)
                # classifier = NormLinear(num_features, num_classes, ctx=ctx, scale=15)
                # self.g_classify.add(classifier)


    def hybrid_forward(self, F, x):
        x = self.base(x)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        # local
        outputs = []
        features = []

        parts_2 = nd.split(x2, axis=2, num_outputs=2)
        for i in range(2):
            part = self.feat[i](parts_2[i])
            if self.pretrained:
                part = self.classify[i](part)
            outputs.append(part)

        parts_3 = nd.split(x3, axis=2, num_outputs=3)
        for i in range(3):
            part = self.feat[i+2](parts_3[i])
            if self.pretrained:
                part = self.classify[i+2](part)
            outputs.append(part)
        
        # global
        g_part_1 = self.g_feat[0](x1)
        g_part_2 = self.g_feat[1](x2)
        g_part_3 = self.g_feat[2](x3)
        features.append(g_part_1)
        features.append(g_part_2)
        features.append(g_part_3)
        if self.pretrained:
            g_part_1 = self.g_classify[0](g_part_1)
            g_part_2 = self.g_classify[1](g_part_2)
            g_part_3 = self.g_classify[2](g_part_3)
        outputs.append(g_part_1)
        outputs.append(g_part_2)
        outputs.append(g_part_3)

        return outputs, features


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