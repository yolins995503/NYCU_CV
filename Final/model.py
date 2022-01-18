# -*- coding: utf-8 -*-
import torch.nn as nn
import torchvision.models as models


class SimCLR(nn.Module):
    def __init__(self, base_encoder, latent_size, for_cifar10=False):
        super(SimCLR, self).__init__()
        # resnet18, resnet34, resnet50, resnet101, resnet152
        # resnext50_32x4d, resnext101_32x8d
        # wide_resnet50_2, wide_resnet101_2
        self.get_backbone(base_encoder)
        repr_size = self.base_encoder.fc.in_features
        if for_cifar10:
            self.base_encoder.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            self.base_encoder.maxpool = nn.Identity()
        self.base_encoder.fc = nn.Identity()
        setattr(self.base_encoder.fc, 'in_features', repr_size)
        setattr(self.base_encoder.fc, 'out_features', repr_size)
        self.projection_head = nn.Sequential(
            nn.Linear(repr_size, repr_size),
            nn.ReLU(inplace=True),
            nn.Linear(repr_size, latent_size))

    def get_backbone(self, base_encoder):
        try:
            exec(f'self.base_encoder = models.{base_encoder}(pretrained=False)')
        except:
            raise ValueError(f'<{base_encoder}> not found')

    def forward(self, img_i, img_j):
        h_i = self.base_encoder(img_i)
        h_j = self.base_encoder(img_j)
        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)
        return h_i, h_j, z_i, z_j
