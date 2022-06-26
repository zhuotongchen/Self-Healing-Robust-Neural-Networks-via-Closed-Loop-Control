#!/usr/bin/env python3
import numpy as np
import os

import torch
import torch.nn as nn

# %%
# Data loader
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.img_labels = labels
        self.img = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# %%
# Generate adversarial samples
from autoattack import AutoAttack

def generate_adv_data(data_loader, models, model_index, perturbation, args):
    eps = args.epsilon
    norm = args.norm
    
    model = models[model_index]
    model.eval()
    correct = 0
    total = 0
    total_batch = len(data_loader)
    
    data_adv = []
    label_set = []
    
    if perturbation == 'autoattack':
        adversary = AutoAttack(model, norm=norm, eps=eps, version='standard')
        
    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        if perturbation == 'None':
            x_adv = inputs
            
        elif perturbation == 'autoattack':
            batch_size = len(labels)
            print('Processing Auto-attack on batch:', i, '/', total_batch)
            x_adv = adversary.run_standard_evaluation(inputs, labels, bs=batch_size)
            
        with torch.no_grad():
            outputs = model(x_adv)
        
        data_adv.append(x_adv)
        label_set.append(labels)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        batch_accu = predicted.eq(labels).sum().item() / labels.shape[0]
        print('Batch ID: {} / {} with Accuracy: {:.2f}'.format(i, total_batch, batch_accu))

    print('Final Accuracy under {}: {:.2f}'.format(perturbation, 100.*correct/total))
    data_adv = torch.cat(data_adv, dim=0)
    label_set = torch.cat(label_set, dim=0)
    torch.save(data_adv.to('cpu'), 'models/dataset/adversarial_data_{}_{}_{}'.format(perturbation, norm, model_index))
    torch.save(label_set.to('cpu'), 'models/dataset/adversarial_labels_{}_{}_{}'.format(perturbation, norm, model_index))

# def get_data_loaders(args):
#     path = 'models/dataset'
#     perturbations = ['autoattack']
#     norms = ['Linf', 'L2', 'L1']
#     train_method = args.train_method

#     data_loaders = []
#     for model_index in range(5):
#         data_loaders_ = []
#         for perturbation in perturbations:
#             for norm in norms:
#                 dirc = os.path.join(path, '{}_{}_{}'.format(perturbation, norm, train_method))
#                 dirc_data = os.path.join(dirc, 'adversarial_data_{}_{}_{}'.format(perturbation, norm, model_index))
#                 dirc_label = os.path.join(dirc, 'adversarial_labels_{}_{}_{}'.format(perturbation, norm, model_index))

#                 dataset = torch.load(dirc_data)
#                 label_set = torch.load(dirc_label)
#                 dataset = CustomImageDataset(images=dataset, labels=label_set)
#                 data_loader = torch.utils.data.DataLoader(dataset,
#                     batch_size=args.batch_size, shuffle=False,
#                     num_workers=args.workers, pin_memory=True)
#                 data_loaders_.append(data_loader)
#         data_loaders.append(data_loaders_)
#     return data_loaders

def get_data_loaders(args):
    path = 'models/dataset'
    norms = ['Linf']

    data_loaders = []
    for model_index in range(5):
        data_loaders_ = []
        for norm in norms:
            dirc_data = os.path.join(path, 'adversarial_data_autoattack_{}_{}'.format(norm, model_index))
            dirc_label = os.path.join(path, 'adversarial_labels_autoattack_{}_{}'.format(norm, model_index))

            dataset = torch.load(dirc_data)
            label_set = torch.load(dirc_label)
            dataset = CustomImageDataset(images=dataset, labels=label_set)
            data_loader = torch.utils.data.DataLoader(dataset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            data_loaders_.append(data_loader)
        data_loaders.append(data_loaders_)
    return data_loaders

def test_data(models, model_index, args):
    norm = args.norm
    perturbation = args.perturbation
    path = 'models/dataset'
    dirc_data = os.path.join(path, 'adversarial_data_autoattack_{}_{}'.format(norm, model_index))
    dirc_label = os.path.join(path, 'adversarial_labels_autoattack_{}_{}'.format(norm, model_index))
    
    dataset = torch.load(dirc_data)
    label_set = torch.load(dirc_label)
    dataset = CustomImageDataset(images=dataset, labels=label_set)
    data_loader = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    model = models[model_index]
    correct = 0
    total = 0
    total_batch = len(data_loader)
    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs)        
            
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        batch_accu = predicted.eq(labels).sum().item() / labels.shape[0]
        # print('Batch ID: {} / {} with Accuracy: {:.2f}'.format(i, total_batch, batch_accu))    
    print('Final Accuracy under {}: {:.2f}'.format(perturbation, 100.*correct/total))


# %%
# Train embedding functions
def lr_schedule(epoch, epochs, lr_max):
    """decrease the learning rate"""
    if epoch / epochs < 0.75:
        return lr_max
    elif epoch / epochs < 0.95:
        return lr_max / 10.
    else:
        return lr_max / 100.

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

def test(data_loader, model):
    model.eval()
    total_loss = 0.
    correct = 0
    total = 0
    total_batch = len(data_loader)

    for i, (data, labels) in enumerate(data_loader):
        data, labels = data.cuda(), labels.cuda()
        with torch.no_grad():
            logits = model(data)
            loss = nn.CrossEntropyLoss(reduction="mean")(logits, labels)
        
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()
    print('Accuracy: {:.3f}, Loss: {:.4f}'.format(100.*correct/total, total_loss/total_batch))



def train_init(train_loader, test_loader, adv_loaders, models, encoder, args):
    epochs = 60
    lr_max = 0.01
    beta = 1.
    optimizer = torch.optim.SGD(encoder.parameters(), lr=lr_max, momentum=0.9, weight_decay=2e-4)
    for epoch in range(epochs):
        selected = np.random.randint(len(models))
        model_selected = models[selected]
        print('Model selected: {}'.format(selected))

        total, correct, loss_total, L2_loss_total = 0, 0, 0., 0.
        lr = lr_schedule(epoch + 1, epochs, lr_max)
        optimizer.param_groups[0].update(lr=lr)
        encoder.train()
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()
            
            recon = encoder(data)
            loss_recon = nn.MSELoss()(data, recon)
            
            logits = model_selected(recon)
            loss_ce = nn.CrossEntropyLoss()(logits, labels)
            
            loss = loss_recon * beta + loss_ce
            loss.backward()
            optimizer.step()

            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            loss_total += loss.item()
            L2_loss_total += (data - recon).norm(p=2, dim=[1, 2, 3]).mean()
            
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], LR: {:.4f}, Loss: {:.4f}, L2: {:.4f}, Acc: {:.2f}' 
                      .format(epoch, epochs, optimizer.param_groups[0]['lr'], loss_total/i,
                                L2_loss_total/i, 100.*correct/total))
                
        dirc = 'models/trained_embedding/ae_init_{}.pt'.format(args.train_method)
        save_model(encoder, optimizer, args, epoch, dirc)
        
        seq_model = nn.Sequential(encoder, model_selected)
        test(test_loader, seq_model)
        test(adv_loaders[selected][0], seq_model)
        

def train_end(train_loader, test_loader, adv_loaders, static_models, encoder_init, encoder_end, args):
    epochs = 60
    lr_max = 0.01
    beta = 10.
    
    state = torch.load('models/trained_embedding/ae_init_{}.pt'.format(args.train_method))
    encoder_init.load_state_dict(state['model'])
    encoder_init.eval()
    
    optimizer = torch.optim.SGD(encoder_end.parameters(), lr=lr_max, momentum=0.9, weight_decay=2e-4)
    for epoch in range(epochs):
        selected = np.random.randint(len(static_models))

        model_selected = static_models[selected]
        print('Model selected: {}'.format(selected))
        
        Linear = model_selected.linear
        model_selected.linear = nn.Sequential()

        total, correct, loss_total, L2_loss_total = 0, 0, 0., 0.
        lr = lr_schedule(epoch + 1, epochs, lr_max)
        optimizer.param_groups[0].update(lr=lr)
        encoder_end.train()
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()
            
            with torch.no_grad():
                recon_init = encoder_init(data)
                hidden = model_selected(recon_init)
                
            recon = encoder_end(hidden)
            loss_recon = nn.MSELoss()(hidden, recon)
            
            logits = Linear(recon)
            loss_ce = nn.CrossEntropyLoss()(logits, labels)
            
            loss = loss_recon * beta + loss_ce
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                logits = Linear(recon)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            loss_total += loss.item()
            L2_loss_total += (hidden - recon).norm(p=2, dim=[1]).mean()
            
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], LR: {:.4f}, Loss: {:.4f}, L2: {:.4f}, Acc: {:.2f}' 
                      .format(epoch, epochs, optimizer.param_groups[0]['lr'], loss_total/i,
                               L2_loss_total/i, 100.*correct/total))
        
        dirc = 'models/trained_embedding/ae_end_{}.pt'.format(args.train_method)
        save_model(encoder_end, optimizer, args, epoch, dirc)
        
        seq_model = nn.Sequential(encoder_init, model_selected, encoder_end, Linear)
        
        test(test_loader, seq_model)
        test(adv_loaders[selected][0], seq_model)
        model_selected.linear = Linear
        
        
        
        
        
        
        