# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def nt_xent_loss(z_i, z_j, temp):
    # z_i, z_j: [b, n]
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    batch_size = z_i.shape[0]
    z = torch.cat((z_i, z_j), dim=0)  # z: [2*b, n]
    sim_mat = torch.exp((z@z.T)/temp)  # sim_mat: [2*b, 2*b]
    mask = torch.ones_like(sim_mat) - torch.eye(2*batch_size, device=device)
    mask = mask.bool()
    sim_mat = sim_mat[mask].reshape(2*batch_size, -1)  # sim_mat: [2*b, 2*b-1]
    pos_sim = torch.exp(torch.sum(z_i*z_j, dim=1)/temp)
    pos_sim = torch.cat((pos_sim, pos_sim), dim=0)  # pos_sim: [2*b]
    loss = -1 * torch.log(pos_sim/sim_mat.sum(dim=1)).mean()
    return loss
