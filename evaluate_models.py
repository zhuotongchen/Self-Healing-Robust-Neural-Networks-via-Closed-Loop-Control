#!/usr/bin/env python3

# evaluate models Table 2 (CIFAR-10 result) from the paper
# to enable the closed-loop control module,
# we need to train embedding functions at input and hidden layers.
# to evaluate the controlled models,
# we need to generate adversarial examples from baseline model,
# finally, we convert baseline models into controlled models
# and pass it into testing.
# In this script the following are implemented:
# 1: generate adversarial examples from baseline models -- set generate_adversarial_dataset
# 2: train embedding function at input layer -- set train_embedding_function_input
# 3: train embedding function at hidden layer -- set train_embedding_function_hidden
# 4: test models (baseline / closed-loop controlled) -- set test_models

import numpy as np
import time, gc

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import random
import argparse

from models.PreAct_rn import preactresnet18, preactresnet34, preactresnet50
from models.Wide_rn import WideResNet28_8, WideResNet34_8
from models.FCN import AutoEncoder, SegNet, FCNs

from control_module import clc_dyn
from control_module_linear import clc_lin_dyn

from control_functions import generate_adv_data, get_data_loaders, test_data, train_init, train_end

from autoattack import AutoAttack



# For reproducibility
torch.manual_seed(999)
np.random.seed(999)
random.seed(999)
torch.cuda.manual_seed_all(999)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic=True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

parser = argparse.ArgumentParser(description='Evaluate models')
parser.add_argument('--train_method', default='standard', type=str, help='training method')
parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
parser.add_argument('--workers', default=2, type=int, help='number_of_workers')

# Select group of models to work
parser.add_argument('--baseline_model', default='standard', type=str, help='select standard or robust baseline models')
parser.add_argument('--encode', default='fcn', type=str, help='select a encoder network')

# adversarial attack setting
parser.add_argument('--epsilon', default=8/255, type=float, help='adversarial training epsilon')
parser.add_argument('--norm', default='Linf', type=str, help='perturbation norm')
parser.add_argument('--perturbation', default='None', type=str, help='perturbation')
parser.add_argument('--wordy', action='store_true', help='Whether to print testing message each batch')

# Control parameters
parser.add_argument('--control', action='store_true', help='Whether to apply control')
parser.add_argument('--pmp_outer_ite', default=3, type=int, help='Choose pmp outer maximum iteration')
parser.add_argument('--pmp_inner_ite', default=10, type=int, help='Choose pmp inner maximum iteration')
parser.add_argument('--regularization', default=0.001, type=int, help='control regularization parameter')

# use pytorch automatic mixed precision
parser.add_argument('--use_amp', action='store_true', help='Whether to automatic mixed precision')

# select a task
# generate adversarial examples dataloder from all baseline models
parser.add_argument('--generate_adversarial_dataset', action='store_true', help='to generate adversarial dataset')

# train embedding function at input layer
parser.add_argument('--train_embedding_function_input', action='store_true', help='to train embedding function at input')

# train embedding function at hidden layer
parser.add_argument('--train_embedding_function_hidden', action='store_true', help='to train embedding function at hidden layer')

# test models
parser.add_argument('--test_models', action='store_true', help='to test models')

args = parser.parse_args()

# %%
# load train and test dataset
def set_loader(opt):
    Train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
        ]), download=True),
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)
    
    Test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)
    return Train_loader, Test_loader

Train_loader, Test_loader = set_loader(args)

# %%
# Set up model
def set_model(opt):
    baseline_method = opt.baseline_model
    static_models = []
    model_names = ['resnet18', 'resnet34', 'resnet50', 'wide_resnet28_8', 'wide_resnet34_8']
    for name in model_names:
        if name == 'resnet18':
            model = preactresnet18().cuda()
        elif name == 'resnet34':
            model = preactresnet34().cuda()
        elif name == 'resnet50':
            model = preactresnet50().cuda()
        elif name == 'wide_resnet28_8':
            model = WideResNet28_8().cuda()
        elif name == 'wide_resnet34_8':
            model = WideResNet34_8().cuda()
        
        state_dir = 'models/trained_models/' + '{}_{}.pt'.format(name, baseline_method)
        state = torch.load(state_dir)
        model.load_state_dict(state)
        model.eval()
        static_models.append(model)
    return static_models

def set_encoder(opt):
    if opt.encode == 'segnet':
        encoder_init = SegNet().cuda()
    elif opt.encode == 'fcn':
        encoder_init = FCNs().cuda()
    
    encoder_end = AutoEncoder(dims=[512, 384, 256]).cuda()
    return encoder_init, encoder_end

static_models = set_model(args)

# %%
# generate adversarial dataset
if args.generate_adversarial_dataset:
    for ind, static_model in enumerate(static_models):
        perturbation = 'autoattack'
        generate_adv_data(Test_loader, static_models, ind, perturbation, args)

    for ind, static_model in enumerate(static_models):
        test_data(static_models, ind, args)

# %%
# train the embedding function at input
if args.train_embedding_function_input:
    adv_loaders = get_data_loaders(args)
    
    encoder_init, _ = set_encoder(args)
    train_init(Train_loader, Test_loader, adv_loaders, static_models, encoder_init, args)

# %%
# train the embedding function at hidden layer
if args.train_embedding_function_hidden:
    adv_loaders = get_data_loaders(args)
    encoder_init, encoder_hidden = set_encoder(args)
    
    train_end(Train_loader, Test_loader, adv_loaders, static_models, encoder_init, encoder_hidden, args)

# %%
# testing function
def testing(data_loader, model, args, wordy=False):
    EPSILON = args.epsilon
    NORM = args.norm
    PERTURBATION = args.perturbation
    BATCH_SIZE = args.batch_size
    
    model.eval()
    correct = 0
    total = 0
    total_batch = len(data_loader)
    
    if PERTURBATION == 'autoattack':
        adversary = AutoAttack(model, norm=NORM, eps=EPSILON, version='standard')
        
    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        if PERTURBATION == 'None':
            x_adv = inputs
            
        elif PERTURBATION == 'autoattack':
            print('Processing Auto-attack on batch:', i, '/', total_batch)
            x_adv = adversary.run_standard_evaluation(inputs, labels, bs=BATCH_SIZE)

        with torch.no_grad():
            outputs = model(x_adv)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if wordy == True:
            batch_accu = predicted.eq(labels).sum().item() / labels.shape[0]
            print('Batch ID: {} / {} with Accuracy: {:.2f}'.format(i, total_batch, batch_accu))

    print('Final Accuracy under {}: {:.2f}'.format(PERTURBATION, 100.*correct/total))


# test closed-loop controlled models
if args.test_models:
    for static_model in static_models:
        if args.control:
            # convert baseline model into a closed-loop controlled model
            model = clc_dyn(static_model, args=args)
        else:
            # test baseline model
            model = static_model
        
        start_time = None
        
        def start_timer():
            global start_time
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.synchronize()
            start_time = time.time()
        
        def end_timer_and_print(local_msg):
            torch.cuda.synchronize()
            end_time = time.time()
            print("\n" + local_msg)
            print("Total execution time = {:.3f} sec".format(end_time - start_time))
            print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))
            
        
        start_timer()
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            testing(Test_loader, model, args, wordy=args.wordy)    
        
        end_timer_and_print("Default precision:")



