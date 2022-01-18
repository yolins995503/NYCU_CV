# -*- coding: utf-8 -*-
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


# vgg definition that conveniently let's you grab the outputs from any layer
def VGG():
    vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU()  # relu5-4
    )
    return vgg


def Decoder():
    decoder = nn.Sequential(  # Sequential,
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )
    return decoder


class VGGEncoder(nn.Module):
    def __init__(self, pretrain_nets):
        super(VGGEncoder, self).__init__()
        enc_layers = list(pretrain_nets.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def forward(self, style_img, content_img=None, with_intermediate=True):
        if with_intermediate:
            feats = self.encode_with_intermediate(style_img)
            return feats[-1], feats
        else:
            content_feat = self.encode(content_img)
            style_feats = self.encode_with_intermediate(style_img)
            return content_feat, style_feats


class AdaINStyleTransfer(nn.Module):
    def __init__(self, vgg_path='./weights/vgg_normalised.pth', decoder_path='./weights/adain_decoder.pth'):
        super(AdaINStyleTransfer, self).__init__()
        self.vgg = VGG()
        self.vgg.load_state_dict(torch.load(vgg_path))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

        self.decoder = Decoder()
        self.decoder.load_state_dict(torch.load(decoder_path))

    def style_transfer_by_image(self, content, style, alpha=1):
        content_feat = self.vgg(content)
        style_feat = self.vgg(style)
        feat = self.adaptive_instance_normalization_by_feat(content_feat, style_feat)
        feat = alpha*feat + (1-alpha)*content_feat
        stylized = self.decoder(feat)
        return stylized

    def style_transfer_by_feat(self, content, style_mean, style_std, alpha=1):
        content_feat = self.vgg(content)
        feat = self.adaptive_instance_normalization_by_stats(content_feat, style_mean, style_std)
        feat = alpha*feat + (1-alpha)*content_feat
        stylized = self.decoder(feat)
        return stylized

    def calc_mean_std(self, feat, eps=1e-6):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert len(size) == 4
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adaptive_instance_normalization_by_feat(self, content_feat, style_feat):
        assert content_feat.size()[:2] == style_feat.size()[:2]
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat-content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat*style_std.expand(size) + style_mean.expand(size)

    def adaptive_instance_normalization_by_stats(self, content_feat, style_mean, style_std):
        assert content_feat.size()[:2] == style_mean.size()[:2]
        assert content_feat.size()[:2] == style_std.size()[:2]
        size = content_feat.size()
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat-content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat*style_std.expand(size) + style_mean.expand(size)
