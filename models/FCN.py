#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

# %%
# SegNet
class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.in_chn = 3
        self.out_chn = 3
        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.vgg16 = models.vgg16_bn(pretrained=True)
  
        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=True)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=True)
  
        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=True)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=True)
  
        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=True)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=True)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=True)
  
        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=True)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=True)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=True)
  
        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=True)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=True)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=True)
        
        self.init_vgg_weigts()
        
        self.MaxDe = nn.MaxUnpool2d(2, stride=2) 
  
        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=True)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=True)
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=True)
  
        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(512, momentum=True)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=True)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(256, momentum=True)
  
        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(256, momentum=True)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=True)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=True)
  
        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=True)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=True)
  
        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=True)
        self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=True)

    def forward(self, x):
        x = F.relu(self.BNEn11(self.ConvEn11(x))) 
        x = F.relu(self.BNEn12(self.ConvEn12(x))) 
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        x = F.relu(self.BNEn21(self.ConvEn21(x))) 
        x = F.relu(self.BNEn22(self.ConvEn22(x))) 
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        x = F.relu(self.BNEn31(self.ConvEn31(x))) 
        x = F.relu(self.BNEn32(self.ConvEn32(x))) 
        x = F.relu(self.BNEn33(self.ConvEn33(x))) 	
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        x = F.relu(self.BNEn41(self.ConvEn41(x))) 
        x = F.relu(self.BNEn42(self.ConvEn42(x))) 
        x = F.relu(self.BNEn43(self.ConvEn43(x))) 	
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        x = F.relu(self.BNEn51(self.ConvEn51(x))) 
        x = F.relu(self.BNEn52(self.ConvEn52(x))) 
        x = F.relu(self.BNEn53(self.ConvEn53(x))) 	
        x, ind5 = self.MaxEn(x)
        size5 = x.size()

        x = self.MaxDe(x, ind5, output_size=size4)
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe51(self.ConvDe51(x)))

        x = self.MaxDe(x, ind4, output_size=size3)
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe41(self.ConvDe41(x)))

        x = self.MaxDe(x, ind3, output_size=size2)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe31(self.ConvDe31(x)))

        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))

        x = self.MaxDe(x, ind1)
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x)
        x = nn.Sigmoid()(x)
        return x
    
    def init_vgg_weigts(self):
        assert self.ConvEn11.weight.size() == self.vgg16.features[0].weight.size()
        self.ConvEn11.weight.data = self.vgg16.features[0].weight.data
        assert self.ConvEn11.bias.size() == self.vgg16.features[0].bias.size()
        self.ConvEn11.bias.data = self.vgg16.features[0].bias.data
        
        assert self.BNEn11.weight.size() == self.vgg16.features[1].weight.size()
        self.BNEn11.weight.data = self.vgg16.features[1].weight.data
        assert self.BNEn11.bias.size() == self.vgg16.features[1].bias.size()
        self.BNEn11.bias.data = self.vgg16.features[1].bias.data

        assert self.ConvEn12.weight.size() == self.vgg16.features[3].weight.size()
        self.ConvEn12.weight.data = self.vgg16.features[3].weight.data
        assert self.ConvEn12.bias.size() == self.vgg16.features[3].bias.size()
        self.ConvEn12.bias.data = self.vgg16.features[3].bias.data
        
        assert self.BNEn12.weight.size() == self.vgg16.features[4].weight.size()
        self.BNEn12.weight.data = self.vgg16.features[4].weight.data
        assert self.BNEn12.bias.size() == self.vgg16.features[4].bias.size()
        self.BNEn12.bias.data = self.vgg16.features[4].bias.data
        
        assert self.ConvEn21.weight.size() == self.vgg16.features[7].weight.size()
        self.ConvEn21.weight.data = self.vgg16.features[7].weight.data
        assert self.ConvEn21.bias.size() == self.vgg16.features[7].bias.size()
        self.ConvEn21.bias.data = self.vgg16.features[7].bias.data
        
        assert self.BNEn21.weight.size() == self.vgg16.features[8].weight.size()
        self.BNEn21.weight.data = self.vgg16.features[8].weight.data
        assert self.BNEn21.bias.size() == self.vgg16.features[8].bias.size()
        self.BNEn21.bias.data = self.vgg16.features[8].bias.data

        assert self.ConvEn22.weight.size() == self.vgg16.features[10].weight.size()
        self.ConvEn22.weight.data = self.vgg16.features[10].weight.data
        assert self.ConvEn22.bias.size() == self.vgg16.features[10].bias.size()
        self.ConvEn22.bias.data = self.vgg16.features[10].bias.data
        
        assert self.BNEn22.weight.size() == self.vgg16.features[11].weight.size()
        self.BNEn22.weight.data = self.vgg16.features[11].weight.data
        assert self.BNEn22.bias.size() == self.vgg16.features[11].bias.size()
        self.BNEn22.bias.data = self.vgg16.features[11].bias.data
        
        assert self.ConvEn31.weight.size() == self.vgg16.features[14].weight.size()
        self.ConvEn31.weight.data = self.vgg16.features[14].weight.data
        assert self.ConvEn31.bias.size() == self.vgg16.features[14].bias.size()
        self.ConvEn31.bias.data = self.vgg16.features[14].bias.data
        
        assert self.BNEn31.weight.size() == self.vgg16.features[15].weight.size()
        self.BNEn31.weight.data = self.vgg16.features[15].weight.data
        assert self.BNEn31.bias.size() == self.vgg16.features[15].bias.size()
        self.BNEn31.bias.data = self.vgg16.features[15].bias.data
        
        assert self.ConvEn32.weight.size() == self.vgg16.features[17].weight.size()
        self.ConvEn32.weight.data = self.vgg16.features[17].weight.data
        assert self.ConvEn32.bias.size() == self.vgg16.features[17].bias.size()
        self.ConvEn32.bias.data = self.vgg16.features[17].bias.data
        
        assert self.BNEn32.weight.size() == self.vgg16.features[18].weight.size()
        self.BNEn32.weight.data = self.vgg16.features[18].weight.data
        assert self.BNEn32.bias.size() == self.vgg16.features[18].bias.size()
        self.BNEn32.bias.data = self.vgg16.features[18].bias.data
        
        assert self.ConvEn33.weight.size() == self.vgg16.features[20].weight.size()
        self.ConvEn33.weight.data = self.vgg16.features[20].weight.data
        assert self.ConvEn33.bias.size() == self.vgg16.features[20].bias.size()
        self.ConvEn33.bias.data = self.vgg16.features[20].bias.data
        
        assert self.BNEn33.weight.size() == self.vgg16.features[21].weight.size()
        self.BNEn33.weight.data = self.vgg16.features[21].weight.data
        assert self.BNEn33.bias.size() == self.vgg16.features[21].bias.size()
        self.BNEn33.bias.data = self.vgg16.features[21].bias.data
        
        assert self.ConvEn41.weight.size() == self.vgg16.features[24].weight.size()
        self.ConvEn41.weight.data = self.vgg16.features[24].weight.data
        assert self.ConvEn41.bias.size() == self.vgg16.features[24].bias.size()
        self.ConvEn41.bias.data = self.vgg16.features[24].bias.data
        
        assert self.BNEn41.weight.size() == self.vgg16.features[25].weight.size()
        self.BNEn41.weight.data = self.vgg16.features[25].weight.data
        assert self.BNEn41.bias.size() == self.vgg16.features[25].bias.size()
        self.BNEn41.bias.data = self.vgg16.features[25].bias.data
        
        assert self.ConvEn42.weight.size() == self.vgg16.features[27].weight.size()
        self.ConvEn42.weight.data = self.vgg16.features[27].weight.data
        assert self.ConvEn42.bias.size() == self.vgg16.features[27].bias.size()
        self.ConvEn42.bias.data = self.vgg16.features[27].bias.data
        
        assert self.BNEn42.weight.size() == self.vgg16.features[28].weight.size()
        self.BNEn42.weight.data = self.vgg16.features[28].weight.data
        assert self.BNEn42.bias.size() == self.vgg16.features[28].bias.size()
        self.BNEn42.bias.data = self.vgg16.features[28].bias.data
        
        assert self.ConvEn43.weight.size() == self.vgg16.features[30].weight.size()
        self.ConvEn43.weight.data = self.vgg16.features[30].weight.data
        assert self.ConvEn43.bias.size() == self.vgg16.features[30].bias.size()
        self.ConvEn43.bias.data = self.vgg16.features[30].bias.data
        
        assert self.BNEn43.weight.size() == self.vgg16.features[31].weight.size()
        self.BNEn43.weight.data = self.vgg16.features[31].weight.data
        assert self.BNEn43.bias.size() == self.vgg16.features[31].bias.size()
        self.BNEn43.bias.data = self.vgg16.features[31].bias.data
        
        assert self.ConvEn51.weight.size() == self.vgg16.features[34].weight.size()
        self.ConvEn51.weight.data = self.vgg16.features[34].weight.data
        assert self.ConvEn51.bias.size() == self.vgg16.features[34].bias.size()
        self.ConvEn51.bias.data = self.vgg16.features[34].bias.data
        
        assert self.BNEn51.weight.size() == self.vgg16.features[35].weight.size()
        self.BNEn51.weight.data = self.vgg16.features[35].weight.data
        assert self.BNEn51.bias.size() == self.vgg16.features[35].bias.size()
        self.BNEn51.bias.data = self.vgg16.features[35].bias.data
        
        assert self.ConvEn52.weight.size() == self.vgg16.features[37].weight.size()
        self.ConvEn52.weight.data = self.vgg16.features[37].weight.data
        assert self.ConvEn52.bias.size() == self.vgg16.features[37].bias.size()
        self.ConvEn52.bias.data = self.vgg16.features[37].bias.data
        
        assert self.BNEn52.weight.size() == self.vgg16.features[38].weight.size()
        self.BNEn52.weight.data = self.vgg16.features[38].weight.data
        assert self.BNEn52.bias.size() == self.vgg16.features[38].bias.size()
        self.BNEn52.bias.data = self.vgg16.features[38].bias.data
        
        assert self.ConvEn53.weight.size() == self.vgg16.features[40].weight.size()
        self.ConvEn53.weight.data = self.vgg16.features[40].weight.data
        assert self.ConvEn53.bias.size() == self.vgg16.features[40].bias.size()
        self.ConvEn53.bias.data = self.vgg16.features[40].bias.data
        
        assert self.BNEn53.weight.size() == self.vgg16.features[41].weight.size()
        self.BNEn53.weight.data = self.vgg16.features[41].weight.data
        assert self.BNEn53.bias.size() == self.vgg16.features[41].bias.size()
        self.BNEn53.bias.data = self.vgg16.features[41].bias.data

# %%
# FCN
class FCNs(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_class = 3
        self.pretrained_net = VGGNet(requires_grad=True)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        
        score = nn.Sigmoid()(score)
        return score  # size=(N, n_class, x.H/1, x.W/1)

from torchvision.models.vgg import VGG

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# %%
# A 2-layer autoencoder
class AutoEncoder(nn.Module):
    def __init__(self, dims=[512, 256, 128]):
        super(AutoEncoder, self).__init__()
        self.E1 = nn.Linear(dims[0], dims[1])
        self.E2 = nn.Linear(dims[1], dims[2])
        self.Ebn1 = nn.BatchNorm1d(dims[1])
        self.Ebn2 = nn.BatchNorm1d(dims[2])
        
        self.D1 = nn.Linear(dims[2], dims[1])
        self.D2 = nn.Linear(dims[1], dims[0])
        
        self.activation = nn.ELU()
        
    def encode(self, x):
        out = self.activation(self.Ebn1(self.E1(x)))
        out = self.activation(self.Ebn2(self.E2(out)))
        return out
    
    def decode(self, x):
        out = self.activation(self.D1(x))
        out = self.D2(out)
        return out
    
    def forward(self, x):
        out = self.encode(x)
        out = self.decode(out)
        return out