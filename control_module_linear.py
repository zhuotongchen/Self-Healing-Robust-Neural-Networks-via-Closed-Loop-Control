#!/usr/bin/env python3

import numpy as np
import os.path
import math

import torch
import torch.nn as nn

from collections import OrderedDict
import os.path
# %%
class empty:
    pass

# Linear control functions
def SVD(A, rank=None):
    num_samples = A.shape[0]
    A = A.view(num_samples, -1)
    
    A_mean = A.mean(dim=0, keepdim=True)
    A -= A_mean
    U, Sigma, V = torch.svd(A)
    A += A_mean
    if rank == None:
        return V, Sigma
    elif isinstance(rank, float):
        var = Sigma
        var_cumsum = torch.cumsum(var, dim=0)
        var_cumsum = var_cumsum / var_cumsum[-1]
        rank = (var_cumsum < rank).sum()
        # print('Rank is: {} / {}'.format(rank, len(Sigma)))
        return V, Sigma, rank
    elif isinstance(rank, int):
        return V, Sigma, rank
    else:
        print('Unknown input type rank')
        return 0
        
class Embedding(nn.Module):
    def __init__(self, dataset, threshold=0.99, reg=0.):
        super(Embedding, self).__init__()
        
        dataset_shape = dataset.shape
        dataset = dataset.view(dataset_shape[0], -1)

        with torch.no_grad():
            data_mean = dataset.mean(dim=0, keepdim=True)
            dataset -= data_mean
            basis, _, rank = SVD(dataset, threshold)
            dim = basis.shape[0]
            diag = torch.ones(dim)
            diag[-rank:] *= (reg / 1 + reg)
            diag = torch.diag(diag).to(dataset.device)
            Proj = basis.mm(diag).mm(basis.t())
            
            diag_orthgonal_compl = torch.ones(dim)
            diag_orthgonal_compl[:rank] *= 0.
            diag_orthgonal_compl = torch.diag(diag_orthgonal_compl).to(dataset.device)
            Proj_orthgonal_compl = basis.mm(diag_orthgonal_compl).mm(basis.t())
            
            dataset += data_mean
            
            self.P = Proj
            self.Proj_compl = Proj_orthgonal_compl
            self.data_mean = data_mean
    
    def projection(self, x):
        x_shape = x.shape
        x = x.view(x_shape[0], -1)
        # x = x - self.data_mean
        x_proj = torch.mm(x, self.P)
        # x_proj = x_proj + self.data_mean
        return x_proj.view(x_shape)
    
    def proj_orth_compl(self, x):
        x_shape = x.shape
        x = x.view(x_shape[0], -1)
        # x = x - self.data_mean
        x_proj = torch.mm(x, self.Proj_compl)
        # x_proj = x_proj + self.data_mean
        return x_proj.view(x_shape)

class clc_lin_dyn(nn.Module):
    def __init__(self, model, inner_ite, outer_ite, eps, threshold, regularization, args=None):
        super(clc_lin_dyn, self).__init__()
        # Run PMP
        self.INNER_ITE = inner_ite
        self.OUTER_ITE = outer_ite
        self.RADIUS = eps
        self.THD = threshold
        self.REG = regularization
        self.opt = args
        
        Linear = model.linear
        model.linear = nn.Sequential()
        self.model = nn.Sequential(model, Linear)
        self.clc_dic = OrderedDict()

    def generate_subspace(self, data_loader):
        # if not isinstance(self.opt, type(None)):
        #     model_name, model_type = self.opt.model_type, self.opt.train_method
        #     save_dir = 'models/lin_control/clc_{}_{}_{}.pt'.format(model_name, model_type, self.REG)
        #     if os.path.isfile(save_dir):
        #         print('Load existing clc dictionary')
        #         self.clc_dic = torch.load(save_dir)
        #         return
        # else:
        #     save_dir = 'models/lin_control/clc.pt'
        data_all = []
        hidden_all = []
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.cuda(), labels.cuda()
                bs = len(labels)
                hiddens = self.model[0](inputs)

                inputs = inputs.view(bs, -1)
                hiddens = hiddens.view(bs, -1)
                
                data_all.append(inputs)
                hidden_all.append(hiddens)
 
        data_all = torch.cat(data_all, dim=0)
        hidden_all = torch.cat(hidden_all, dim=0)
        
        embedding_in = Embedding(data_all, self.THD, reg=self.REG)
        embedding_hid = Embedding(hidden_all, self.THD, self.REG)

        self.clc_dic['embedding'] = [embedding_in, embedding_hid]
        
        # num_samples_used = data_all.shape[0]
        # print('Number of Samples used: ', num_samples_used)
        
        # torch.save(self.clc_dic, save_dir)

    def normalize(self, x):
        ndims = len(x.shape[1:])
        if self.PMP_NORM == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
        elif self.PMP_NORM == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        return x / (t.view(-1, *([1] * ndims)) + 1e-12)

    def initialize_cont(self, x):
        x_shape = x.shape
        if len(x_shape) < 4:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            recons_in = self.clc_dic['embedding'][0].projection(x)
            cont_in = recons_in - x
        cont_in.detach_()
        return [cont_in]

    def Maximize_Hamilton(self, x, control, encode, adjoint, F=None):
        bs = x.shape[0]
        
        optimizer = torch.optim.SGD([control], lr=(self.RADIUS / 2), momentum=0.9, weight_decay=0.)
        for ii in range(self.INNER_ITE):
            optimizer.zero_grad()
            
            control.requires_grad_()
            x_cont = x + control
            
            # Reconstruction loss
            x_orth_compl = encode.proj_orth_compl(x_cont)
            loss_recon = (x_orth_compl**2).view(bs, -1).sum(dim=1).mean()
            
            # control regularization loss
            loss_regu = (control ** 2).view(bs, -1).sum(dim=1).mean()
            
            # Total loss
            loss = loss_recon / 2. + loss_regu * self.REG / 2.

            # Compute Hamiltonian at t
            x_cont_t_plus1 = F(x_cont)
            H = (x_cont_t_plus1 * adjoint).view(bs, -1).sum(dim=1).mean()
            H = H + loss

            H.backward()
            optimizer.step()
        
        control.detach_()
        if control.grad is not None:
            control.grad.zero_()
        return control
    
    def forward(self, x):
        bs = x.shape[0]
        x_ = x.clone()
        x_.detach_()
        if x_.grad is not None:
            x_.grad.zero_()
        
        control = self.initialize_cont(x_)

        with torch.enable_grad():
            for ii in range(self.OUTER_ITE):
                # Forward propagation
                x_cont = x_ + control[0]
                hiddens = self.model[0](x_cont)

                # Hidden state control has analytic solution
                # Since the adjoint P = 0
                hiddens_cont = self.clc_dic['embedding'][1].projection(hiddens)
                
                # Compute reconstruction loss at hidden
                hiddens_orth_compl = self.clc_dic['embedding'][1].proj_orth_compl(hiddens_cont)
                loss_recon = (hiddens_orth_compl**2).view(bs, -1).sum(dim=1).mean()

                # Compute control regularization loss
                cont_at_hidden = hiddens_cont - hiddens
                loss_regu = (cont_at_hidden ** 2).view(bs, -1).sum(dim=1).mean()
                
                # Total loss
                loss = loss_recon / 2. + loss_regu * self.REG / 2.
                
                loss.backward()
                with torch.no_grad():
                    adjoint = self.model[0].layer_one_out.grad * (-1.)
                # Maximizing Hamiltonian at inputs
                control[0] = self.Maximize_Hamilton(x_, control[0], self.clc_dic['embedding'][0], adjoint=adjoint, F=self.model[0].conv1)
            
        x_cont = x + control[0]
        hiddens = self.model[0](x_cont)
        hiddens_cont = self.clc_dic['embedding'][1].projection(hiddens)
        outputs = self.model[1](hiddens_cont)
        return outputs


