# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:52:09 2020

@author: adds0
"""
"""
Generator Code below was inspired from https://github.com/jalola/improved-wgan-pytorch"""

from torch import nn
import torch
from utils.utilities import *
DIM=64
OUTPUT_DIM=256*256*3
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True,  stride = 1, bias = True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias = bias)

    def forward(self, input):
        output = self.conv(input)
        return output
class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size,input_height,output_width,output_depth) for t_t in spl]
        output = torch.stack(stacks,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size,output_height,output_width,output_depth)
        output = output.permute(0, 3, 1, 2)
        return output
class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=DIM):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        if resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)

        else:
            raise Exception('invalid resample value')


        if resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = UpSampleConv(input_dim, output_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(output_dim, output_dim, kernel_size = kernel_size)

        else:
            raise Exception('invalid resample value')

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output
class GoodGenerator(nn.Module):
    def __init__(self, dim=64,output_dim=3*256*256):
        super(GoodGenerator, self).__init__()

        self.dim = dim

        self.ln1 = nn.Sequential(nn.Linear(128, 4*4*8*self.dim),
                                 nn.ReLU())
        self.rb1 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'up')
        self.rb2 = ResidualBlock(8*self.dim, 4*self.dim, 3, resample = 'up')
        self.rb3 = ResidualBlock(4*self.dim, 2*self.dim, 3, resample = 'up')
        self.rb4 = ResidualBlock(2*self.dim, 1*self.dim, 3, resample = 'up')
        self.rb5 = ResidualBlock(1*self.dim, 32, 3, resample = 'up')
        self.rb6 = ResidualBlock(32, 16, 3, resample = 'up')
        self.bn  = nn.BatchNorm2d(16)

        self.conv1 = MyConvo2d(16, 3, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        #output = self.ln1(input)
        output = self.ln1(input.contiguous())
        output = output.view(input.size(0),512, 4,4)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        
        output = self.rb5(output)
        output = self.rb6(output)
        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)

        return output
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
# =============================================================================
#             ## Critic Network
# =============================================================================
            
class Discriminator(torch.nn.Module):
    def __init__(self, image_channel_size=3, channel_size=16):
        super().__init__()
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size
        
        self.main_module = nn.Sequential(

            nn.Conv2d(in_channels=image_channel_size, out_channels=channel_size, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(channel_size),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.3),

            nn.Conv2d(in_channels=channel_size, out_channels=channel_size*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(channel_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.3),


            nn.Conv2d(in_channels=channel_size*2, out_channels=channel_size*4, kernel_size=4, stride=2, padding=1),
            #nn.PReLU(),
            nn.InstanceNorm2d(channel_size*4),
            #nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.3),
            nn.Conv2d(in_channels=channel_size*4, out_channels=channel_size*8, kernel_size=4, stride=2, padding=1),
            #nn.PReLU(),
            nn.InstanceNorm2d(channel_size*8),
            ##nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3),
            
            nn.Conv2d(in_channels=channel_size*8, out_channels=channel_size*16, kernel_size=4, stride=2, padding=1),#conv2d
            #nn.PReLU(),
            nn.InstanceNorm2d(channel_size*16),
            # nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            
            nn.Conv2d(in_channels=channel_size*16, out_channels=channel_size*32, kernel_size=4, stride=2, padding=1),
            #nn.PReLU()
            nn.InstanceNorm2d(channel_size*32),
            # nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5))

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=channel_size*32, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)
    
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
