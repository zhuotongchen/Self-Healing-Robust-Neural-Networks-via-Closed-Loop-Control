#!/usr/bin/env python3

# This file trains baseline classifiers
# for both standard training and adversarial training (TRADES)
# Baseline models include resnet and wide resnet
# dataset is cifar10

import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable

import random
import argparse

from models.PreAct_rn import preactresnet18, preactresnet34, preactresnet50
from models.model_resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from models.Wide_rn import WideResNet28_8, WideResNet34_8

# For reproducibility
torch.manual_seed(999)
np.random.seed(999)
random.seed(999)
torch.cuda.manual_seed_all(999)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic=True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

parser = argparse.ArgumentParser(description='Train neural networks')
# Model training parameters
parser.add_argument('--model_type', type=str, default="resnet18", help="decide which network to use")
parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train')
parser.add_argument('--weight_decay', default=2e-4, type=float, help='l2 regularization')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--train_method', default='standard', type=str, help='training method')
# Adversarial training parameters
parser.add_argument('--epsilon', default=8./255, type=float, help='adversarial training epsilon')
parser.add_argument('--step_size', default=2./255, type=float, help='adversarial training step_size')
parser.add_argument('--num_steps', default=10, type=int, help='adversarial training num_steps')
parser.add_argument('--rand_init', default=False, type=bool, help='pgd rand initialization')
parser.add_argument('--number_of_workers', default=2, type=int, help='number_of_workers')

# TARDES parameters
parser.add_argument('--beta', default=6.0, type=float, help='regularization, i.e., 1/lambda in TRADES')
args = parser.parse_args()


# %%
# Learning rate schedule for training
def lr_schedule(epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    return lr

# Save traind model
def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

# %%
# load train and test dataset
def set_loader(opt):
    Train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
        ]), download=True),
        batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.number_of_workers, pin_memory=True)
    
    Test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.number_of_workers, pin_memory=True)
    return Train_loader, Test_loader

Train_loader, Test_loader = set_loader(args)

# %%
# Set up model
def set_model(opt):
    if opt.model_type == 'resnet18':
        return preactresnet18().cuda()
    elif opt.model_type == 'resnet34':
        return preactresnet34().cuda()
    elif opt.model_type == 'resnet50':
        return preactresnet50().cuda()
    elif opt.model_type == 'resnet20':
        return resnet20().cuda()
    elif opt.model_type == 'resnet32':
        return resnet32().cuda()
    elif opt.model_type == 'resnet44':
        return resnet44().cuda()
    elif opt.model_type == 'resnet56':
        return resnet56().cuda()    
    elif opt.model_type == 'resnet110':
        return resnet110().cuda()  
    
    elif opt.model_type == 'wide_resnet28_8':
        return WideResNet28_8().cuda()
    elif opt.model_type == 'wide_resnet34_8':
        return WideResNet34_8().cuda()
    
    else:
        print('Warning: invalid model type !')

# %%
# Standard training with SGD
def train(train_loader, optimizer, model, epoch, args):
    train_loss = 0
    train_correct = 0
    train_total = 0
    total_step = len(train_loader)
    model.train()

    # Set learning rate
    lr = lr_schedule(epoch + 1)
    optimizer.param_groups[0].update(lr=lr)
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = nn.CrossEntropyLoss(reduction="mean")(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        if (i+1) % 100 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], LR: {:.4f}, Loss: {:.4f}, Acc: {:.3f}' 
                               .format(epoch+1, args.epochs, i+1, total_step,
                                       optimizer.param_groups[0]['lr'], train_loss/total_step, 100.*train_correct/train_total)) 


# %%
# TRADES loss
def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # beta = torch.tensor(beta).cuda()
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = torch.optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss, logits

# %%
# Adversarial training with SGD
def PGD(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)

        model.zero_grad()
        with torch.enable_grad():
            loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = torch.autograd.Variable(x_adv, requires_grad=False)
    return x_adv

def train_adv(train_loader, optimizer, model, epoch, args):
    train_loss = 0
    train_correct = 0
    train_total = 0
    total_step = len(train_loader)
    
    # Set model to train mode
    model.train()
    
    # Set learning rate
    lr = lr_schedule(epoch + 1)
    optimizer.param_groups[0].update(lr=lr)
    for i, (data, labels) in enumerate(train_loader):
        data, labels = data.cuda(), labels.cuda()
        optimizer.zero_grad()

        loss, logits = trades_loss(model=model,
                    x_natural=data,
                    y=labels,
                    optimizer=optimizer,
                    step_size=args.step_size,
                    epsilon=args.epsilon,
                    perturb_steps=args.num_steps,
                    beta=args.beta,
                    distance='l_inf')

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = logits.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        if (i+1) % 100 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], LR: {:.4f}, Loss: {:.4f}, Acc: {:.3f}' 
                               .format(epoch+1, args.epochs, i+1, total_step,
                                       optimizer.param_groups[0]['lr'], train_loss/total_step, 100.*train_correct/train_total)) 

def test(data_loader, model, perturbation, epsilon=None, step_size=None, num_steps=None):
    model.eval()
    total_loss = 0.
    correct = 0
    total = 0
    total_batch = len(data_loader)

    for i, (data, labels) in enumerate(data_loader):
        data, labels = data.cuda(), labels.cuda()
        if perturbation == 'None':
            data = data

        elif perturbation == 'pgd':
            data = PGD(model, data, labels, epsilon, step_size, num_steps,
                                  loss_fn="cent", category="Madry", rand_init=True)
        with torch.no_grad():
            logits = model(data)
            loss = nn.CrossEntropyLoss(reduction="mean")(logits, labels)
        
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()

    print('Accuracy under {}: {:.3f}, Loss: {:.4f}'.format(perturbation, 100.*correct/total, total_loss/total_batch))

def main():
    model = set_model(args)
    
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum, 
                                weight_decay=args.weight_decay)

    for num_epochs in range(args.epochs):
        if args.train_method == 'standard':
            train(Train_loader, optimizer, model, num_epochs, args)
        elif args.train_method == 'robust':
            train_adv(Train_loader, optimizer, model, num_epochs, args)
            test(Test_loader, model, perturbation='pgd', epsilon=args.epsilon,
                 step_size=args.step_size, num_steps=args.num_steps)
            
        test(Test_loader, model, perturbation='None')
        
        direct = 'models/trained_models/{}_{}.pt'.format(args.model_type, args.train_method)
        torch.save(model.state_dict(), direct)

if __name__ == '__main__':
    main()