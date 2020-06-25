import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

class Discriminator80x80InstNorm(nn.Module):
    def __init__(self, device='cpu', pretrained=False, patchSize=[64, 64], input_nc=3):
        super(Discriminator80x80InstNorm, self).__init__()
        self.device = device
        self.input_nc = input_nc
        self.patchSize = patchSize
        self.outputSize = [patchSize[0]/16, patchSize[1]/16]
                        
        self.discriminator = nn.Sequential(         
            # 128-->60
            nn.Conv2d(self.input_nc, 64, kernel_size=5, padding=0, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),            
            
            # 60-->33
            nn.Conv2d(64, 128, kernel_size=5, padding=0, stride=2, bias=False),
            nn.InstanceNorm2d(128, momentum=0.001, affine=False, track_running_stats=False), 
            nn.LeakyReLU(0.2, inplace=True),
            # 33->
            nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=2, bias=False),
            nn.InstanceNorm2d(256, momentum=0.001, affine=False, track_running_stats=False), 
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(256, 512, kernel_size=3, padding=0, stride=2, bias=False),
            nn.InstanceNorm2d(512, momentum=0.001, affine=False, track_running_stats=False), 
            nn.LeakyReLU(0.2, inplace=True),            
            # final classification for 'real(1) vs. fake(0)'
            nn.Conv2d(512, 1, kernel_size=1, padding=0, stride=1, bias=True),
        )        
        
    def forward(self, X):        
        return self.discriminator(X)

class Discriminator80x80InstNormDilation(nn.Module):
    # same as Discriminator80x80InstNorm except the kernel size of last layer is changed to 3x3
    # used to test receptive field
    def __init__(self, device='cpu', dialate_size=1, pretrained=False, patchSize=[64, 64], input_nc=3):
        super(Discriminator80x80InstNormDilation, self).__init__()
        self.device = device
        self.input_nc = input_nc
        self.patchSize = patchSize
        self.outputSize = [patchSize[0]/16, patchSize[1]/16]
        self.dialate_size = dialate_size
                        
        self.discriminator = nn.Sequential(         
            # 128-->60
            nn.Conv2d(self.input_nc, 64, kernel_size=5, padding=0, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),            
            
            # 60-->33
            nn.Conv2d(64, 128, kernel_size=5, padding=0, stride=2, bias=False),
            nn.InstanceNorm2d(128, momentum=0.001, affine=False, track_running_stats=False), 
            nn.LeakyReLU(0.2, inplace=True),
            # 33->
            nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=2, bias=False),
            nn.InstanceNorm2d(256, momentum=0.001, affine=False, track_running_stats=False), 
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(256, 512, kernel_size=3, padding=0, stride=2, bias=False),
            nn.InstanceNorm2d(512, momentum=0.001, affine=False, track_running_stats=False), 
            nn.LeakyReLU(0.2, inplace=True),            
            # final classification for 'real(1) vs. fake(0)'
            nn.Conv2d(512, 1, kernel_size=3, padding=0, stride=1, bias=True, dilation=self.dialate_size),
        )        
        
    def forward(self, X):        
        return self.discriminator(X)

class Discriminator5121520InstNorm(nn.Module):
    def __init__(self, device='cpu', pretrained=False, patchSize=[64, 64], input_nc=3):
        super(Discriminator5121520InstNorm, self).__init__()
        self.device = device
        self.input_nc = input_nc
        self.patchSize = patchSize
        self.outputSize = [patchSize[0]/16, patchSize[1]/16]
                        
        self.discriminator = nn.Sequential(
            # 128-->60
            nn.Conv2d(self.input_nc, 256, kernel_size=3, padding=0, stride=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 60-->33
            nn.Conv2d(256, 128, kernel_size=3, padding=0, stride=1, bias=False),
            nn.InstanceNorm2d(128, momentum=0.001, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 33->
            nn.Conv2d(128, 64, kernel_size=3, padding=0, stride=1, bias=False),
            nn.InstanceNorm2d(64, momentum=0.001, affine=False, track_running_stats=False), 
            nn.LeakyReLU(0.2, inplace=True),

            # final classification for 'real(1) vs. fake(0)'
            nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=True),
        )
        
    def forward(self, X):        
        return self.discriminator(X)

class DiscriminatorGlobalLocal(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, bbox_size = 64, conv_dim=64, c_dim=5, repeat_num_global=6, repeat_num_local=5, nc=3):
        super(DiscriminatorGlobalLocal, self).__init__()

        maxFilt = 512 if image_size==128 else 128
        globalLayers = []
        globalLayers.append(nn.Conv2d(nc, conv_dim, kernel_size=4, stride=2, padding=1,bias=False))
        globalLayers.append(nn.LeakyReLU(0.2, inplace=True))

        localLayers = []
        localLayers.append(nn.Conv2d(nc, conv_dim, kernel_size=4, stride=2, padding=1, bias=False))
        localLayers.append(nn.LeakyReLU(0.2, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num_global):
            globalLayers.append(nn.Conv2d(curr_dim, min(curr_dim*2,maxFilt), kernel_size=4, stride=2, padding=1, bias=False))
            globalLayers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = min(curr_dim * 2, maxFilt)

        curr_dim = conv_dim
        for i in range(1, repeat_num_local):
            localLayers.append(nn.Conv2d(curr_dim, min(curr_dim * 2, maxFilt), kernel_size=4, stride=2, padding=1, bias=False))
            localLayers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = min(curr_dim * 2, maxFilt)

        k_size_local = int(bbox_size/ np.power(2, repeat_num_local))
        k_size_global = int(image_size/ np.power(2, repeat_num_global))

        self.mainGlobal = nn.Sequential(*globalLayers)
        self.mainLocal = nn.Sequential(*localLayers)

        # FC 1 for doing real/fake
        # self.fc1 = nn.Linear(curr_dim*(k_size_local**2+k_size_global**2), 1, bias=False)
        self.fc1 = nn.Linear(10880, 1, bias=False)

        # FC 2 for doing classification only on local patch
        if c_dim > 0:
            self.fc2 = nn.Linear(curr_dim*(k_size_local**2), c_dim, bias=False)
        else:
            self.fc2 = None

    def forward(self, x, boxImg, classify=False):
        bsz = x.size(0)
        h_global = self.mainGlobal(x)
        h_local = self.mainLocal(boxImg)
        h_append = torch.cat([h_global.view(bsz,-1), h_local.view(bsz,-1)], dim=-1)
        out_rf = self.fc1(h_append)
        out_cls = self.fc2(h_local.view(bsz,-1)) if classify and (self.fc2 is not None) else None
        return out_rf.squeeze(), out_cls, h_append