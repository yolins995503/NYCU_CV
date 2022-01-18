# -*- coding: utf-8 -*-
import os
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%Y%m%d-%H:%M:%S',
)


class LinearEvaluationProtocol(nn.Module):
    def __init__(self, encoder, args, logger=None, save_cpt=False, cpt_dir=None):
        super(LinearEvaluationProtocol, self).__init__()
        self.encoder = encoder
        self.class_num = args.class_num
        self.linear_classifier = nn.Linear(self.encoder.fc.in_features, args.class_num)
        self.to(device)
        self.encoder = self.encoder.eval()
        self.logger = logger
        self.save_cpt = save_cpt
        self.cpt_dir = cpt_dir

        self.best_top1_acc = 0
        self.best_top5_acc = 0

        self.init_optimizer(args.eval_lr, args.eval_weight_decay)

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)
        y = self.linear_classifier(h)
        return y

    def init_optimizer(self, lr=1e-3, weight_decay=1e-6):
        self.lr = lr
        self.weight_decay = weight_decay
        self.opt = optim.Adam(self.linear_classifier.parameters(), lr=lr, weight_decay=weight_decay)

    def set_data(self, train_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def fit(self, epochs):
        logging.info('Start training linear evaluation protocol')
        epoch_pbar = tqdm(range(epochs))
        for epoch in epoch_pbar:
            self._epoch_fit(epoch)
            with torch.no_grad():
                top1_acc, top5_acc = self._epoch_eval()

            if self.logger is not None:
                self.logger.add_scalar('linear_eval_protocol/val/top1_acc', top1_acc, epoch+1)
                self.logger.add_scalar('linear_eval_protocol/val/top5_acc', top5_acc, epoch+1)

            epoch_pbar.set_description(f'epoch: {epoch+1}/{epochs}, top1 acc:{top1_acc:.5f}, top5 acc:{top5_acc:.5f}')
            if self.best_top1_acc < top1_acc:
                self.best_top1_acc = top1_acc
            if self.best_top5_acc < top5_acc:
                self.best_top5_acc = top5_acc
            logging.info(f'best top1 acc: {self.best_top1_acc:.5f}, best top5 acc: {self.best_top5_acc:.5f}')
            if self.save_cpt:
                torch.save(self.linear_classifier.state_dict(), os.path.join(self.cpt_dir, f'epoch{epoch+1}_model.cpt'))
                torch.save(self.opt.state_dict(), os.path.join(self.cpt_dir, f'epoch{epoch+1}_optim.cpt'))

        logging.info('Finish training of linear evaluation protocol')
        logging.info(f'best top1 acc: {self.best_top1_acc:.5f}, best top5 acc: {self.best_top5_acc:.5f}')
        return self.best_top1_acc, self.best_top5_acc

    def _epoch_fit(self, epoch):
        pbar = tqdm(self.train_dataloader)
        avg_top1_acc = 0
        avg_top5_acc = 0
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            y_pred = self.forward(x)
            loss = F.cross_entropy(y_pred, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            y_pred_top1 = y_pred.topk(1)[1]
            y_pred_top5 = y_pred.topk(5)[1]
            
            top1_acc = (y_pred_top1==y.reshape(-1, 1)).float().mean()
            top5_acc = (y_pred_top5==y.reshape(-1, 1).repeat(1, 5)).float().sum(-1).mean()

            avg_top1_acc += top1_acc
            avg_top5_acc += top5_acc

            pbar.set_description(f'loss: {loss.item():.5f}, top1 acc:{top1_acc.item():.5f}, top5 acc:{top5_acc.item():.5f}')

        avg_top1_acc /= len(self.train_dataloader)
        avg_top5_acc /= len(self.train_dataloader)
        if self.logger is not None:
            self.logger.add_scalar('linear_eval_protocol/train/top1_acc', avg_top1_acc, epoch+1)
            self.logger.add_scalar('linear_eval_protocol/train/top5_acc', avg_top5_acc, epoch+1)

    def _epoch_eval(self):
        pbar = tqdm(self.test_dataloader)

        total_top1_acc = []
        total_top5_acc = []

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            y_pred = self.forward(x)
            loss = F.cross_entropy(y_pred, y)

            y_pred_top1 = y_pred.topk(1)[1]
            y_pred_top5 = y_pred.topk(5)[1]

            top1_acc = (y_pred_top1==y.reshape(-1, 1))
            total_top1_acc.append(top1_acc)
            top5_acc = (y_pred_top5==y.reshape(-1, 1).repeat(1, 5)).sum(-1)
            total_top5_acc.append(top5_acc)

        total_top1_acc = torch.cat(total_top1_acc).float().mean()
        total_top5_acc = torch.cat(total_top5_acc).float().mean()

        return total_top1_acc.item(), total_top5_acc.item()
