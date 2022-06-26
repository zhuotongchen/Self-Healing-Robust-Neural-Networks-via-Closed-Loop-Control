#!/usr/bin/env python3

import numpy as np

import torch
import torch.nn as nn

from models.FCN import AutoEncoder, SegNet, FCNs
# %%
class clc_dyn(nn.Module):
    def __init__(self, static_model, args, inner_ite=None, outer_ite=None, lr_factor=None, regularization=None):
        super(clc_dyn, self).__init__()
        # Run PMP
        if inner_ite is None:
            self.Max_Inner_ite = args.pmp_inner_ite
        else:
            self.Max_Inner_ite = inner_ite

        if outer_ite is None:
            self.Max_Outer_ite = args.pmp_outer_ite
        else:
            self.Max_Outer_ite = outer_ite        

        if lr_factor is None:
            self.PMP_LR = args.epsilon / 2.
        else:
            self.PMP_LR = args.epsilon / lr_factor        
        
        if regularization is None:
            self.const = args.regularization
        else:
            self.const = regularization      
        
        self.PMP_RADIUS = torch.tensor(args.epsilon).cuda()
        self.PMP_NORM = args.norm
        self.args = args

        # Static model being controlled
        Linear = static_model.linear
        static_model.linear = nn.Sequential()
        self.model = nn.Sequential(static_model, Linear)
        
        # Load encoder for inputs
        if args.encode == 'segnet':
            encoder_init = SegNet().cuda()
        elif args.encode == 'fcn':
            encoder_init = FCNs().cuda()
        elif args.encode == 'ae':
            encoder_init = DeAutoEncoder(channels=[3, 64]).cuda()
        state = torch.load('models/trained_embedding/ae_init_standard.pt')
        encoder_init.load_state_dict(state['model'])
        encoder_init.eval()
        
        # Load encoder for hidden
        encoder_end = AutoEncoder(dims=[512, 384, 256]).cuda()
        state_end = torch.load('models/trained_embedding/ae_end_standard.pt')
        encoder_end.load_state_dict(state_end['model'])
        encoder_end.eval()
    
        self.encode = [encoder_init, encoder_end]
    
    def eval_threshold(self, data_loader):
        loss_total1 = 0.
        loss_total2 = 0.
        with torch.no_grad():
            for i, (data, labels) in enumerate(data_loader):
                data, labels = data.cuda(), labels.cuda()
                recon = self.encode[0](data)
                hiddens = self.model[0](recon)
                
                recon_hiddens = self.encode[1](hiddens)
                
                loss_total1 += nn.MSELoss()(data, recon).item()
                # loss_total2 += nn.MSELoss()(hiddens, recon_hiddens).item()
                loss_total2 += (hiddens - recon_hiddens).norm(p=2, dim=1).mean()
        self.thd1 = loss_total1 / len(data_loader)
        self.thd2 = loss_total2 / len(data_loader)
        print('Threshold inp: {:.4f}, Threshold hid: {:.4f}'.format(self.thd1, self.thd2))
    
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
            recons_in = self.encode[0](x)
            hiddens = self.model[0](recons_in)
            recons_hid = self.encode[1](hiddens)
            
            cont_in = recons_in - x
            cont_hid = recons_hid - hiddens
            
        cont_in.detach_()
        cont_hid.detach_()
        return [cont_in, cont_hid]
    
    def Maximize_Hamilton(self, x, control, encode, adjoint=None, F=None):
        bs = x.shape[0]
        ndims = len(x.shape[1:])
        
        optimizer = torch.optim.SGD([control], lr=self.PMP_LR, momentum=0.9, weight_decay=self.const)
        for ii in range(self.Max_Inner_ite):
            optimizer.zero_grad()
            
            control.requires_grad_()
            x_cont = x + control
            recons = encode(x_cont)
            
            # Compute running loss at t
            loss_recon = ((x_cont - recons)**2).view(bs, -1).sum(dim=1).mean()
            # loss_regu = (control ** 2).view(bs, -1).sum(dim=1).mean()
            # loss = loss_recon + self.const * loss_regu
            loss = loss_recon
            # Compute Hamiltonian at t
            if adjoint is not None:
                x_cont_t_plus1 = F(x_cont)
                H = (x_cont_t_plus1 * adjoint).view(bs, -1).sum(dim=1).mean()
                H = H + loss
            else:
                H = loss
            H.backward()
            optimizer.step()
                        
        # with torch.no_grad():
        #     if self.PMP_NORM == 'Linf':
        #         control = torch.min(torch.max(control, -self.PMP_RADIUS), self.PMP_RADIUS)
        #     elif self.PMP_NORM == 'L2':
        #         cont_l2norm = control.view(bs, -1).norm(p=2, dim=-1).view(-1, *([1] * ndims))
        #         control = self.normalize(control) * torch.min(self.PMP_RADIUS * torch.ones_like(cont_l2norm).detach(), cont_l2norm)

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
            for ii in range(self.Max_Outer_ite):
                x_cont = x_ + control[0]
                hiddens = self.model[0](x_cont)
                
                # Maximizing Hamiltonian at hiddens
                hiddens_ = hiddens.clone()
                hiddens_.detach_()
                control[1] = self.Maximize_Hamilton(hiddens_, control[1], self.encode[1], adjoint=None, F=None)
                
                # Backward propagation for adjoint variable
                hiddens_cont = hiddens + control[1]
                recons = self.encode[1](hiddens_cont)
                
                loss_recon = ((recons - hiddens_cont)**2).view(bs, -1).sum(dim=1).mean()
                loss_regu = (control[1] ** 2).view(bs, -1).sum(dim=1).mean()
                
                loss = loss_recon + self.const * loss_regu
                
                loss.backward()
                with torch.no_grad():
                    adjoint = self.model[0].layer_one_out.grad * (-1.)
                # Maximizing Hamiltonian at inputs
                control[0] = self.Maximize_Hamilton(x_, control[0], self.encode[0], adjoint=adjoint, F=self.model[0].conv1)
            
        x_cont = x + control[0]
        hiddens = self.model[0](x_cont)
        hiddens_cont = hiddens + control[1]
        outputs = self.model[1](hiddens_cont)
        return outputs
    



        
        
        
        
        
        


