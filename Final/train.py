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
from loss import nt_xent_loss
from augmentation import ColorDistortion
from linear_eval import LinearEvaluationProtocol
from dataset import ContrastiveImageNetDataset, ContrastiveCIFAR10Dataset, \
                    ContrastiveTinyImageNetDataset, ImageNetDataset, \
                    ImageNet64Dataset, TinyImageNetDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
	datefmt='%Y%m%d-%H:%M:%S',
)


def load_args():
    parser = ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str, default='imagenet', help='imagenet, cifar10')
    parser.add_argument('--train_set', type=str)
    parser.add_argument('--val_set', type=str)
    parser.add_argument('--meta_path', type=str, default='./ImageNet_meta/meta.mat')
    parser.add_argument('--gt_path', type=str, default='./ImageNet_meta/ILSVRC2012_validation_ground_truth.txt')
    parser.add_argument('--img_size', type=int, default=224)

    parser.add_argument('--cpt_interval', type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--cpt_dir', type=str, default='cpts')

    # model
    parser.add_argument('--base_encoder', type=str, default='resnet50',
                        help='resnet18, resnet34, resnet50, resnet101, resnet152, \
                              resnext50_32x4d, resnext101_32x8d, \
                              wide_resnet50_2, wide_resnet101_2')
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--nt_xent_temp', type=float, default=0.1)
    parser.add_argument('--adain', action='store_true', default=False)
    parser.add_argument('--adain_alpha', type=float, default=0.5)
    parser.add_argument('--adain_preserve_mode', action='store_true', default=False)
    parser.add_argument('--adain_size', type=int, default=128)

    # training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)

    # evaluate
    parser.add_argument('--linear_eval', action='store_true', default=False)
    parser.add_argument('--linear_eval_epoch', type=int, default=100, 
                        help='which trainining epoch run linear eval')
    parser.add_argument('--eval_dataset', type=str, default='cifar10', 
                        help='imagenet64, imagenet, cifar10, tinyimagenet')
    parser.add_argument('--eval_train_set', type=str)
    parser.add_argument('--eval_val_set', type=str)
    parser.add_argument('--eval_epochs', type=int, default=90)
    parser.add_argument('--eval_lr', type=float, default=1e-3)
    parser.add_argument('--eval_weight_decay', type=float, default=1e-6)

    # continue training
    parser.add_argument('--continue_training', action='store_true', default=False)
    parser.add_argument('--continue_epochs', type=int)
    parser.add_argument('--continue_iter', type=int)
    parser.add_argument('--opt_path', type=str)
    parser.add_argument('--cpt_path', type=str)

    # exp name
    args = parser.parse_args()
    exp_name = f'{args.dataset}_'
    if args.continue_training:
        exp_name += f'continue_E{args.continue_epochs}_iter_{args.continue_iter}_'
    exp_name += '{}_img_size-{}_z_size-{}_temp-{}_E-{}_lr-{}_b1-{}_b2-{}_bs-{}'.format(
                args.base_encoder, args.img_size, args.latent_size, args.nt_xent_temp,
                args.num_epochs, args.lr, args.beta_1, args.beta_2,
                args.batch_size)
    if args.adain:
        exp_name += f'_adain-{args.adain_alpha}'
        if args.adain_preserve_mode:
            exp_name += f'_preserve-{args.adain_size}'
    if args.linear_eval:
        exp_name += f'_eval-{args.eval_dataset}'

    args.log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(args.log_dir, exist_ok=True)
    args.cpt_dir = os.path.join(args.cpt_dir, exp_name)
    os.makedirs(args.cpt_dir, exist_ok=True)
    return args


def train(
        model, logger, train_loader, valid_loader, optimizer, args, 
        linear_eval_train_loader, linear_eval_val_loader):
    color_distortion = ColorDistortion(args.img_size, use_adain=args.adain, alpha=args.adain_alpha,
                                       adain_preserve_mode=args.adain_preserve_mode, 
                                       adain_size=args.adain_size)
    batch_done = 0
    if args.continue_training:
        batch_done += args.continue_iter
    pbar_epoch = tqdm(range(args.num_epochs))
    for epoch in pbar_epoch:
        if args.continue_training:
            if epoch < args.continue_epochs:
                continue
        model.train()
        epoch_loss = 0
        pbar_batch = tqdm(train_loader)
        for batch_idx, (imgs_i, imgs_j) in enumerate(pbar_batch):
            imgs_i = imgs_i.to(device)
            imgs_j = imgs_j.to(device)
            with torch.no_grad():
                imgs_i = color_distortion(imgs_i)
                imgs_j = color_distortion(imgs_j)

            optimizer.zero_grad()
            h_i, h_j, z_i, z_j = model(imgs_i, imgs_j)
            loss = nt_xent_loss(z_i, z_j, args.nt_xent_temp)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            batch_done += 1
            logger.add_scalar('train/loss', loss.item(), batch_done)
            pbar_batch.set_description('epoch: {}/{}, batch: {}/{}, loss: {:.5f}'.format(
                epoch+1, args.num_epochs, batch_idx+1, len(train_loader), loss.item()))

            if batch_done%args.cpt_interval == 0:
                save_path = os.path.join(args.cpt_dir, f'E-{epoch+1}_iter-{batch_done}')
                logging.info(f'Save model cpt @ {save_path}')
                torch.save(model.state_dict(), save_path+'_model.cpt')
                torch.save(optimizer.state_dict(), save_path+'_optim.cpt')

        epoch_loss /= len(train_loader)
        logging.info('----------------------------')
        pbar_epoch.set_description(f'epoch: {epoch+1}/{args.num_epochs}, avg. loss: {epoch_loss:.5f}')
        val_loss = evaluate(model, valid_loader)
        logger.add_scalar('val/loss', val_loss, batch_done)
        logging.info('============================')
        if args.linear_eval and (epoch+1)%args.linear_eval_epoch==0:
            logging.info('Initialize linear evaluation protocol')
            linear_evaluator = LinearEvaluationProtocol(model.base_encoder, args)
            logging.info('Initialize dataloader of linear evaluation protocol')
            linear_evaluator.set_data(linear_eval_train_loader, linear_eval_val_loader)
            logging.info('Initialize optimizer of linear evaluation protocol')
            top1_acc, top5_acc = linear_evaluator.fit(args.eval_epochs)
            logger.add_scalar('eval/top1_acc', top1_acc, batch_done)
            logger.add_scalar('eval/top5_acc', top5_acc, batch_done)

        save_path = os.path.join(args.cpt_dir, f'E-{epoch+1}_iter-{batch_done}')
        logging.info(f'Save model cpt @ {save_path}')
        torch.save(model.state_dict(), save_path+'_model.cpt')
        torch.save(optimizer.state_dict(), save_path+'_optim.cpt')


def evaluate(model, val_loader):
    model.eval()
    pbar_eval = tqdm(val_loader)
    correct = 0
    val_loss = 0
    for batch_idx, (imgs_i, imgs_j) in enumerate(pbar_eval):
        imgs_i = imgs_i.to(device)
        imgs_j = imgs_j.to(device)
        with torch.no_grad():
            h_i, h_j, z_i, z_j = model(imgs_i, imgs_j)
            loss = nt_xent_loss(z_i, z_j, args.nt_xent_temp)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss


if __name__ == '__main__':
    args = load_args()

    if args.dataset=='imagenet':
        train_dataset = ContrastiveImageNetDataset(args.train_set, args.img_size)
        val_dataset = ContrastiveImageNetDataset(args.val_set, args.img_size)
    elif args.dataset=='cifar10':
        train_dataset = ContrastiveCIFAR10Dataset(args.train_set, train=True)
        val_dataset = ContrastiveCIFAR10Dataset(args.val_set, train=False)
    elif args.dataset=='tinyimagenet':
        train_dataset = ContrastiveTinyImageNetDataset(args.train_set)
        val_dataset = ContrastiveTinyImageNetDataset(args.val_set)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.linear_eval:
        if args.eval_dataset == 'imagenet64':
            linear_eval_train_dataset = ImageNet64Dataset(
                root_dir=args.eval_train_set, mode='train')
            linear_eval_val_dataset = ImageNet64Dataset(
                root_dir=args.eval_val_set, mode='val')
            args.class_num = linear_eval_train_dataset.class_num
        elif args.eval_dataset == 'imagenet':
            linear_eval_train_dataset = ImageNetDataset(
                root_dir=args.train_set, meta_path=args.meta_path,
                img_size=args.img_size, mode='train')
            linear_eval_val_dataset = ImageNetDataset(
                root_dir=args.val_set, meta_path=args.meta_path, gt_path=args.gt_path,
                img_size=args.img_size, mode='val')
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
    else:
        linear_eval_train_loader = None
        linear_eval_val_loader = None

    if args.dataset=='imagenet':
        model = SimCLR(args.base_encoder, args.latent_size, for_cifar10=False)
    elif args.dataset=='cifar10':
        model = SimCLR(args.base_encoder, args.latent_size, for_cifar10=True)
    elif args.dataset=='tinyimagenet':
        model = SimCLR(args.base_encoder, args.latent_size, for_cifar10=False)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
    logger = SummaryWriter(args.log_dir)

    if args.continue_training:
        logging.info('Continue training! Loading checkpoint...')
        logging.info(f'Loading {args.cpt_path}...')
        model.load_state_dict(torch.load(args.cpt_path))
        logging.info(f'Loading {args.opt_path}...')
        optimizer.load_state_dict(torch.load(args.opt_path))

    train(
        model, logger, train_loader, val_loader, optimizer, args,
        linear_eval_train_loader, linear_eval_val_loader)
