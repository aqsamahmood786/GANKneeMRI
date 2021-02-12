# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:09:20 2020

@author: adds0
"""
# =============================================================================
#                        Import Libraries
# =============================================================================
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.optim as optim
from torch import autograd
import time
import os
from torchvision import utils
import numpy as np 
import imageio
from utils.utilities import *
from utils.logger import Logger
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
from torch.nn import Parameter


# set the computation device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# =============================================================================
#                                  ## CNN based Generator to generate images of size 64*64
# =============================================================================
class Generator(torch.nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        
        self.main_module = nn.Sequential(
            # Z latent vector 128
            nn.ConvTranspose2d(in_channels=128, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            #512*4*4

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            
            #256*8*8
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            
            #128*16*16
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            #64*32*32
        
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1))
            #3*64*64
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    


# =============================================================================
#                                 #Discriminator
# =============================================================================

class Discriminator(torch.nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        
        self.main_module = nn.Sequential(

            
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),#conv2d
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5))

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0,dilation=1, groups=1))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)
    
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
# =============================================================================
#                                # ResNet Based Generator to generate 64*64 
# =============================================================================
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
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=64):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        #self.relu1 = nn.ReLU()
        #self.relu2 = nn.ReLU()
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            #TODO: ????
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size = kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = UpSampleConv(input_dim, output_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(output_dim, output_dim, kernel_size = kernel_size)
        elif resample == None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size = kernel_size)
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
    def __init__(self, dim=64,output_dim=3*64*64):
        super(GoodGenerator, self).__init__()

        self.dim = dim

        self.ln1 = nn.Sequential(nn.Linear(128, 4*4*8*self.dim),
                                 nn.ReLU())
        self.rb1 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'up')
        self.rb2 = ResidualBlock(8*self.dim, 4*self.dim, 3, resample = 'up')
        self.rb3 = ResidualBlock(4*self.dim, 2*self.dim, 3, resample = 'up')
        self.rb4 = ResidualBlock(2*self.dim, 1*self.dim, 3, resample = 'up')
        self.bn  = nn.BatchNorm2d(64)

        self.conv1 = MyConvo2d(64, 3, 3)
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
        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        return output
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

# =============================================================================
#            Trying diffferent objective function
# =============================================================================
# =============================================================================
#                    # Calclate divergence gradient penalty
# =============================================================================
def calculate_div_GP(real_data,fake_data, output_real, output_fake):
    
    real_grad_outputs = torch.ones((real_data.size(0),1,1,1), requires_grad=True).to(device)
    real_grad = autograd.grad(output_real, real_data, real_grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
    fake_grad_outputs = torch.ones((fake_data.size(0),1,1,1), requires_grad=True).to(device)
    fake_grad = autograd.grad(output_fake, fake_data, fake_grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
    div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

    return div_gp
# =============================================================================
# # Train discriminator with WGAN Div objective function
# =============================================================================
def train_discriminator(optimizer, real_data, fake_data):
    b_size = real_data.size(0)
    

    
    output_real = discriminator(real_data)
    output_real_loss = output_real.mean()

    
    output_fake = discriminator(fake_data)
    output_fake_loss = output_fake.mean()
    gradient_penalty_div = calculate_div_GP(real_data,fake_data, output_real, output_fake)


    
    d_loss = output_fake_loss-output_real_loss + gradient_penalty_div
    d_loss.backward(retain_graph=True)#retain_graph=True
    wasserstein_distance = output_fake_loss-output_real_loss 

    
    optimizer.step()
    discriminator_loss = d_loss.item()
    
    return [discriminator_loss, output_real_loss, output_fake_loss, wasserstein_distance,gradient_penalty_div]


# =============================================================================
#             ## Calculate Gradient Plenty
# =============================================================================

def calculate_gradient_penalty(batch_size, real_data, fake_data):
    eta = torch.FloatTensor(batch_size,1,1,1).normal_(0,1)
    eta = eta.expand(batch_size, real_data.size(1), real_data.size(2), real_data.size(3))
    eta = eta.to(device)
        
    interpolated = eta * real_data + ((1 - eta) * fake_data)
    interpolated = interpolated.to(device)
        
    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)
        
    # calculate probability of interpolated examples
    
    prob_interpolated = discriminator(interpolated)
        
        # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                              create_graph=True, retain_graph=True)[0]
        
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty

# =============================================================================
#                          #Critic with WGAN GP objective function
# =============================================================================

def train_discriminator_GP(optimizer, real_data, fake_data):
    b_size = real_data.size(0)
    

    
    output_real = discriminator(real_data)
    output_real_loss = output_real.mean()

    
    output_fake = discriminator(fake_data)
    output_fake_loss = output_fake.mean()

    
    gradient_penalty = calculate_gradient_penalty(b_size, real_data.data, fake_data.data)
    

    
    d_loss = output_fake_loss-output_real_loss + gradient_penalty
    d_loss.backward(retain_graph=True)
    wasserstein_distance = output_real_loss - output_fake_loss 

    
    optimizer.step()
    discriminator_loss = d_loss.item()
    
    return [discriminator_loss, output_real_loss, output_fake_loss]
# =============================================================================
#                 ##Hyperparameter values for Tunning
# =============================================================================
learning_rates = [0.01,0.001,0.0001,0.00001, 0.00005]
#Choosing different learning rate for generator and discriminator
generator_learning_rate = 0.00001
critic_learning_rate =0.000005
