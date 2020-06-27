# importing libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image


class GramMatrix(nn.Module):
    def forward(self, y):
        (b, ch, h, w) = y.size()
        # b - batch size(=1)
        # ch - number of feature maps
        # (h, w) - dimensions of a f. map
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


class Inspiration(nn.Module):
    def __init__(self, c, b=1):
        super(Inspiration, self).__init__()

        # b is equal to 1 or input mini_batch
        self.weight = nn.Parameter(torch.Tensor(1, c, c), requires_grad=True)
        # non-parameter buffer
        self.G = Variable(torch.Tensor(b, c, c), requires_grad=True)
        self.C = c
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0.0, 0.02)

    def set_target(self, target):
        self.G = target

    def forward(self, x):
        # input x is a 3D feature map
        self.P = torch.bmm(self.weight.expand_as(self.G), self.G)

        return torch.bmm(self.P.transpose(1, 2).expand(x.size(0), self.C, self.C),
                         x.view(x.size(0), x.size(1), -1)).view_as(x)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'N x ' + str(self.C) + ')'


# some basic layers, with reflectance padding
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()

        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)

        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()

        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)

        self.reflection_padding = int(np.floor(kernel_size / 2))
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)

        if self.reflection_padding != 0:
            x = self.reflection_pad(x)

        out = self.conv2d(x)

        return out


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()

        self.expansion = 4
        self.downsample = downsample

        if self.downsample is not None:
            self.residual_layer = nn.Conv2d(inplanes, planes * self.expansion,
                                            kernel_size=1, stride=stride)

        conv_block = []
        conv_block += [norm_layer(inplanes, track_running_stats=True),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
        conv_block += [norm_layer(planes, track_running_stats=True),
                       nn.ReLU(inplace=True),
                       ConvLayer(planes, planes, kernel_size=3, stride=stride)]
        conv_block += [norm_layer(planes, track_running_stats=True),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        if self.downsample is not None:
            residual = self.residual_layer(x)
        else:
            residual = x
        return residual + self.conv_block(x)


class UpBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super(UpBottleneck, self).__init__()

        self.expansion = 4
        self.residual_layer = UpsampleConvLayer(inplanes, planes * self.expansion,
                                                kernel_size=1, stride=1, upsample=stride)

        conv_block = []
        conv_block += [norm_layer(inplanes, track_running_stats=True),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
        conv_block += [norm_layer(planes, track_running_stats=True),
                       nn.ReLU(inplace=True),
                       UpsampleConvLayer(planes, planes, kernel_size=3, stride=1, upsample=stride)]
        conv_block += [norm_layer(planes, track_running_stats=True),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.residual_layer(x) + self.conv_block(x)


# the MSG Net
class Net(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=6, gpu_ids=None):
        super(Net, self).__init__()

        if gpu_ids is None:
            gpu_ids = []
        self.gpu_ids = gpu_ids
        self.gram = GramMatrix()

        block = Bottleneck
        upblock = UpBottleneck
        expansion = 4

        model1 = []
        model1 += [ConvLayer(input_nc, 64, kernel_size=7, stride=1),
                   norm_layer(64, track_running_stats=True),
                   nn.ReLU(inplace=True),
                   block(64, 32, 2, 1, norm_layer),
                   block(32 * expansion, ngf, 2, 1, norm_layer)]
        self.model1 = nn.Sequential(*model1)

        model = []
        self.ins = Inspiration(ngf * expansion)
        model += [self.model1]
        model += [self.ins]

        for i in range(n_blocks):
            model += [block(ngf * expansion, ngf, 1, None, norm_layer)]

        model += [upblock(ngf * expansion, 32, 2, norm_layer),
                  upblock(32 * expansion, 16, 2, norm_layer),
                  norm_layer(16 * expansion, track_running_stats=True),
                  nn.ReLU(inplace=True),
                  ConvLayer(16 * expansion, output_nc, kernel_size=7, stride=1)]

        self.model = nn.Sequential(*model)

    def set_target(self, xs):
        F = self.model1(xs)
        G = self.gram(F)
        self.ins.set_target(G)

    def forward(self, inp):
        return self.model(inp)


def tensor_load_rgb_image(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')

    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)

    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgb_image(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()

    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgr_image(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgb_image(tensor, filename, cuda)


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch
