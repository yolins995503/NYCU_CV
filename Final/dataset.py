# -*- coding: utf-8 -*-
import os
import pickle
import logging
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
import scipy.io as sio

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%Y%m%d-%H:%M:%S',
)


def read_meta(path):
    meta = sio.loadmat(path)
    meta = meta['synsets']
    columns = meta.dtype.names
    table = []
    for row in meta:
        row = row[0]
        row = (
            row[0][0,0], row[1][0], row[2][0],
            row[3][0], row[4][0,0], row[5],
            row[6][0,0], row[7][0,0])
        table.append(row)
    df = pd.DataFrame(table, columns=columns)
    df.sort_values(by=['ILSVRC2012_ID'], inplace=True)
    return df


class ContrastiveImageNetDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        assert os.path.isdir(root_dir), f'{root_dir} is not a valid directory'
        self.img_list = []
        logging.info(f'Finding images under {root_dir}...')
        self.img_list = glob(os.path.join(root_dir, '**/*.JPEG'), recursive=True)
        logging.info(f'{len(self.img_list)} images found')
        self.spatial = transforms.Compose([
            transforms.RandomResizedCrop(size=img_size),
            transforms.RandomHorizontalFlip()
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        path = self.img_list[idx]
        img = to_tensor(Image.open(path).convert('RGB'))
        img_i = self.spatial(img)
        img_j = self.spatial(img)
        return img_i, img_j


class ContrastiveCIFAR10Dataset(CIFAR10):
    def __init__(self, root, train, download=True):
        self.root = root
        self.train = train

        if download:
            self.download()

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        self.spatial = transforms.Compose([
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = to_tensor(self.data[idx])
        img_i = self.spatial(img)
        img_j = self.spatial(img)
        return img_i, img_j


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, meta_path, gt_path=None, img_size=224, mode='train'):
        assert os.path.isdir(root_dir), f'{root_dir} is not a valid directory'
        assert os.path.isfile(meta_path), f'{meta_path} is not a valid path'

        self.class_num = 1000
        self.img_list = []
        logging.info(f'Finding images under {root_dir}...')
        self.img_list = glob(os.path.join(root_dir, '**/*.JPEG'), recursive=True)
        self.img_list = sorted(self.img_list)
        logging.info(f'{len(self.img_list)} images found')

        meta = read_meta(meta_path)
        class_id = 0
        self.ilsvrc_to_class = {}
        self.wn_to_class = {}
        for (ilsvrc_id, wn_id) in zip(meta['ILSVRC2012_ID'], meta['WNID']):
            self.ilsvrc_to_class[str(ilsvrc_id)] = class_id
            self.wn_to_class[wn_id] = class_id
            class_id += 1
            if class_id >= 1000:
                break

        self.labels = []
        if mode == 'train':
            for img in self.img_list:
                wnid = img.split('/')[-1].split('_')[0]
                self.labels.append(self.wn_to_class[wnid])
        elif mode == 'val':
            ilsvrc_ids = open(gt_path, 'r').readlines()
            self.labels = [self.ilsvrc_to_class[ilsvrc_id.strip()] for ilsvrc_id in ilsvrc_ids]
            assert len(self.labels) == len(self.img_list)
        else:
            raise ValueError(f'Unknown mode <{mode}>')

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=img_size),
            transforms.RandomHorizontalFlip()
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        path = self.img_list[idx]
        img = self.transform(Image.open(path).convert('RGB'))
        label = self.labels[idx]
        return img, torch.tensor(label).long()


class ImageNet64Dataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        '''
        use npz version of ImageNet64 
        '''
        assert os.path.isdir(root_dir), f'{root_dir} is not a valid directory'
        self.class_num = 1000
        logging.info(f'Initialize ImageNet64 {mode} set')
        self.images = None
        self.labels = None
        if mode == 'train':
            pbar = tqdm(os.listdir(root_dir))
            for npz_file in pbar:
                pbar.set_description(f'Loading... {npz_file}')
                npz_file = np.load(os.path.join(root_dir, npz_file))
                if self.images is None:
                    self.images = npz_file['data'].reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1)
                else:
                    self.images = np.concatenate((self.images, npz_file['data'].reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1)))
                if self.labels is None:
                    self.labels = npz_file['labels']
                else:
                    self.labels = np.concatenate((self.labels, npz_file['labels']))

            self.images = np.concatenate(self.images)
            self.labels = np.concatenate(self.labels)
            assert len(self.labels) == len(self.images)

        elif mode == 'val':
            npz_file = np.load(os.path.join(root_dir, 'val_data.npz'))
            self.images = npz_file['data'].reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1)
            self.lebels = npz_file['labels']
            assert len(self.labels) == len(self.images)
        else:
            raise ValueError(f'Unknown mode <{mode}>')

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip()
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])
        label = self.labels[idx]
        return img, torch.tensor(label).long()


class ContrastiveTinyImageNetDataset(Dataset):
    def __init__(self, root_dir, preload=True):
        assert os.path.isdir(root_dir), f'{root_dir} is not a valid directory'
        self.preload = preload
        logging.info(f'Finding images under {root_dir}...')
        self.img_list = glob(os.path.join(root_dir, '**/*.JPEG'), recursive=True)
        logging.info(f'{len(self.img_list)} images found')
        if preload:
            self.images = []
            logging.info('Start preload...')
            for img in tqdm(self.img_list):
                self.images.append(Image.open(img).convert('RGB'))
        self.spatial = transforms.Compose([
            transforms.RandomResizedCrop(size=64, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip()
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.preload:
            img = to_tensor(self.images[idx])
        else:
            path = self.img_list[idx]
            img = to_tensor(Image.open(path).convert('RGB'))
        img_i = self.spatial(img)
        img_j = self.spatial(img)
        return img_i, img_j


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', preload=True):
        assert os.path.isdir(root_dir), f'{root_dir} is not a valid directory'
        self.preload = preload
        self.class_num = 200

        wn_ids = open(os.path.join(root_dir, '../wnids.txt'), 'r').readlines()
        class_id = 0
        self.wn_to_class = {}
        for wn_id in wn_ids:
            self.wn_to_class[wn_id.strip()] = class_id
            class_id += 1
        assert self.class_num == class_id

        self.labels = []
        logging.info(f'Finding images under {root_dir}...')
        if mode == 'train':
            self.img_list = glob(os.path.join(root_dir, '**/*.JPEG'), recursive=True)
            for img in self.img_list:
                wnid = img.split('/')[-1].split('_')[0]
                self.labels.append(self.wn_to_class[wnid])
        elif mode == 'val':
            val_annot = open(os.path.join(root_dir, 'val_annotations.txt'), 'r').readlines()
            self.fn_to_wn = {}
            for annot in val_annot:
                tmp = annot.split()
                self.fn_to_wn[tmp[0].strip()] = tmp[1].strip()
            self.img_list = glob(os.path.join(root_dir, '**/*.JPEG'), recursive=True)
            for img in self.img_list:
                img_fn = img.split('/')[-1]
                self.labels.append(self.wn_to_class[self.fn_to_wn[img_fn]])
        else:
            raise ValueError(f'Unknown mode <{mode}>')
        logging.info(f'{len(self.img_list)} images found')

        if preload:
            self.images = []
            logging.info('Start preload...')
            for img in tqdm(self.img_list):
                self.images.append(Image.open(img).convert('RGB'))

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip()
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.preload:
            img = self.transform(self.images[idx])
        else:
            path = self.img_list[idx]
            img = self.transform(Image.open(path).convert('RGB'))
        label = self.labels[idx]
        return img, torch.tensor(label).long()
