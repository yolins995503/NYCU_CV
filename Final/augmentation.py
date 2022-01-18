# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from PIL import Image

import kornia.augmentation as K

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import resize, to_tensor

from AdaIN import AdaINStyleTransfer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class StyleTransfer(nn.Module):
    def __init__(self, use_style_img=False, alpha=0.5):
        super(StyleTransfer, self).__init__()
        self.stylizer = AdaINStyleTransfer().to(device)
        self.stylizer = self.stylizer.eval()
        self.use_style_img = use_style_img
        self.alpha = alpha
        if use_style_img:
            self.styles = []
            for style_img in os.listdir('style'):
                img = Image.open(os.path.join('style', style_img)).convert('RGB')
                img = resize(to_tensor(img), 512)
                self.styles.append(img)

    def forward(self, imgs):
        if self.use_style_img:
            styles = random.choices(self.styles, k=imgs.shape[0])
            styles = torch.stack(styles, dim=0)
            return self.stylizer.style_transfer_by_image(imgs, styles, alpha=self.alpha)
        else:
            style_mean = torch.rand(imgs.shape[0], 512, 1, 1).to(device)
            style_std = torch.rand(imgs.shape[0], 512, 1, 1).to(device)
            return self.stylizer.style_transfer_by_feat(imgs, style_mean, style_std, alpha=self.alpha)


def random_sigma():
    sigma = np.random.uniform(0.1, 2)
    return (sigma, sigma)


class ColorDistortion(nn.Module):
    def __init__(self, img_size, s=1, use_adain=True, alpha=0.5, 
                adain_preserve_mode=False, adain_size=128):
        super(ColorDistortion, self).__init__()
        self.img_size = img_size
        self.adain_preserve_mode = adain_preserve_mode
        self.use_adain = use_adain
        self.adain_size = adain_size
        if use_adain:
            self.transform = transforms.Compose([
                StyleTransfer(alpha=alpha)
            ])
        else:
            kernel_size = int(0.1*img_size) + int(int(0.1*img_size)%2==0)
            self.transform = transforms.Compose([
                K.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s, p=0.8),
                K.RandomGrayscale(p=0.2),
                K.GaussianBlur(
                    kernel_size=(kernel_size, kernel_size),
                    sigma=random_sigma(), p=0.5)
            ])

    def forward(self, img):
        if self.use_adain and self.adain_preserve_mode and self.img_size<self.adain_size:
            img = resize(img, self.adain_size)
            img = self.transform(img)
            img = resize(img, self.img_size)
            return img
        else:
            return self.transform(img)
