# -*- coding: utf-8 -*-
import os
import logging
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import SimCLR
from linear_eval import LinearEvaluationProtocol
from dataset import ImageNetDataset, ImageNet64Dataset, TinyImageNetDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%Y%m%d-%H:%M:%S',
)


def load_args():
    parser = ArgumentParser()
    # SimCLR
    parser.add_argument('--base_encoder', type=str, default='resnet50',
                        help='resnet18, resnet34, resnet50, resnet101, resnet152, \
                              resnext50_32x4d, resnext101_32x8d, \
                              wide_resnet50_2, wide_resnet101_2')
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--cpt_path', type=str)

    # data
    parser.add_argument('--eval_img_size', type=int, default=224)
    parser.add_argument('--eval_dataset', type=str, default='cifar10', 
                        help='imagenet64, imagenet, cifar10, tinyimagenet')
    parser.add_argument('--eval_train_set', type=str)
    parser.add_argument('--eval_val_set', type=str)
    parser.add_argument('--eval_meta_path', type=str, default='./ImageNet_meta/meta.mat')
    parser.add_argument('--eval_gt_path', type=str, default='./ImageNet_meta/ILSVRC2012_validation_ground_truth.txt')

    # linear evaluatation protocol
    parser.add_argument('--eval_epochs', type=int, default=90)
    parser.add_argument('--eval_lr', type=float, default=1e-3)
    parser.add_argument('--eval_weight_decay', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--save_cpt', action='store_true', default=False)
    parser.add_argument('--cpt_dir', type=str, help='{cpt_folder}/{clr_exp_name}', required=True)
    parser.add_argument('--log_dir', type=str, help='{log_folder}/{clr_exp_name}', required=True)

    args = parser.parse_args()
    suffix = f'_linear_eval_{args.eval_dataset}_E-{args.eval_epochs}_lr-{args.eval_lr}_wd-{args.eval_weight_decay}_bs-{args.batch_size}'
    args.cpt_dir += suffix
    args.log_dir += suffix
    os.makedirs(args.cpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    return args


def train(model, logger, args, linear_eval_train_loader, linear_eval_val_loader):
    logging.info('Initialize linear evaluation protocol')
    linear_evaluator = LinearEvaluationProtocol(model.base_encoder, args, logger, args.save_cpt, args.cpt_dir)
    logging.info('Initialize dataloader of linear evaluation protocol')
    linear_evaluator.set_data(linear_eval_train_loader, linear_eval_val_loader)
    logging.info('Initialize optimizer of linear evaluation protocol')
    best_top1_acc, best_top5_acc = linear_evaluator.fit(args.eval_epochs)
    return best_top1_acc, best_top5_acc


if __name__ == '__main__':
    args = load_args()

    if args.eval_dataset == 'imagenet64':
        linear_eval_train_dataset = ImageNet64Dataset(
            root_dir=args.eval_train_set, mode='train')
        linear_eval_val_dataset = ImageNet64Dataset(
            root_dir=args.eval_val_set, mode='val')
        args.class_num = linear_eval_train_dataset.class_num
    elif args.eval_dataset == 'imagenet':
        linear_eval_train_dataset = ImageNetDataset(
            root_dir=args.eval_train_set, meta_path=args.eval_meta_path,
            img_size=args.eval_img_size, mode='train')
        linear_eval_val_dataset = ImageNetDataset(
            root_dir=args.eval_val_set, meta_path=args.eval_meta_path, gt_path=args.eval_gt_path,
            img_size=args.eval_img_size, mode='val')
        args.class_num = linear_eval_train_dataset.class_num
    elif args.eval_dataset == 'cifar10':
        tf = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                             torchvision.transforms.RandomHorizontalFlip()])
        linear_eval_train_dataset = torchvision.datasets.CIFAR10(args.eval_train_set, train=True, download=True, transform=tf)
        linear_eval_val_dataset = torchvision.datasets.CIFAR10(args.eval_val_set, train=False, download=True, transform=tf)
        args.class_num = 10
    elif args.eval_dataset == 'tinyimagenet':
        linear_eval_train_dataset = TinyImageNetDataset(root_dir=args.eval_train_set, mode='train')
        linear_eval_val_dataset = TinyImageNetDataset(root_dir=args.eval_val_set, mode='val')
        args.class_num = linear_eval_train_dataset.class_num
    else:
        raise ValueError('Unknown evaluation dataset')

    linear_eval_train_loader = DataLoader(
        linear_eval_train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    linear_eval_val_loader = DataLoader(
        linear_eval_val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SimCLR(args.base_encoder, args.latent_size)
    model.load_state_dict(torch.load(args.cpt_path))
    model = model.to(device)
    logger = SummaryWriter(args.log_dir)
    train(model, logger, args, linear_eval_train_loader, linear_eval_val_loader)
