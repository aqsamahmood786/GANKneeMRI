# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:55:09 2020

@author: adds0
"""

# =============================================================================
#                        Import Libraries
# =============================================================================
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch import autograd
import time
import os
from torchvision import utils
import numpy as np 
import imageio
from utils.utilities import create_noise,get_infinite_batches
from utils.logger import Logger
from models.resnet_G_and_C import GoodGenerator,Discriminator


class WGANTrainer:
    def __init__(self, generator,discriminator,generator_optimizer,discriminator_optimizer,device,lambda_term = 10):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.lambda_term = lambda_term
        self.device = device
        
    # =============================================================================
    #             ## Calculate Gradient Plenty
    # =============================================================================
    
    def calculate_gradient_penalty(self,batch_size, real_data, fake_data):
        eta = torch.FloatTensor(batch_size,1,1,1).normal_(0,1)
        eta = eta.expand(batch_size, real_data.size(1), real_data.size(2), real_data.size(3))
        eta = eta.to(self.device)
            
        interpolated = eta * real_data + ((1 - eta) * fake_data)
        interpolated = interpolated.to(self.device)
            
        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)
            
        # calculate probability of interpolated examples
        
        prob_interpolated = self.discriminator(interpolated)
            
            # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                                  create_graph=True, retain_graph=True)[0]
            
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty
    
    # =============================================================================
    #                          #Train Discriminator
    # =============================================================================
    
    def train_discriminator(self,optimizer, real_data, fake_data):
        b_size = real_data.size(0)
        
    
        
        output_real = self.discriminator(real_data)
        output_real_loss = output_real.mean()
    
        
        output_fake = self.discriminator(fake_data)
        output_fake_loss = output_fake.mean()
    
        gradient_penalty = self.calculate_gradient_penalty(b_size, real_data.data, fake_data.data)
        
    
        
        d_loss = output_fake_loss-output_real_loss + gradient_penalty
        d_loss.backward(retain_graph=True)
        wasserstein_distance = output_fake_loss-output_real_loss 
    
        
        optimizer.step()
        discriminator_loss = d_loss.item()
        
        return [discriminator_loss, output_real_loss, output_fake_loss, wasserstein_distance,gradient_penalty]
    
    # =============================================================================
    #                           #Train Generator
    # =============================================================================
        
    def train_generator(self,optimizer, noise):
        fake_data = self.generator(noise)
        bch_size = fake_data.size(0)
        
    
        
        g_output = self.discriminator(fake_data)
        g_loss = -g_output.mean()
        g_loss.backward()
    
        optimizer.step()
        g_cost = g_loss.item()
        return g_cost
    # =============================================================================
    #                 # Save  and Load Model
    # =============================================================================
    def save_model(self,results_dir):
        gen_pkl = os.path.join(results_dir,'generator.pkl' )
        disc_pkl = os.path.join(results_dir,'discriminator.pkl' )
        
        torch.save(self.generator.state_dict(), gen_pkl)
        torch.save(self.discriminator.state_dict(), disc_pkl)
        print('Models save to {} & {}'.format(gen_pkl,disc_pkl))
        
    def load_model(self,D_model_filename, G_model_filename, resultDir):
        D_model_path = os.path.join(resultDir, D_model_filename)
        G_model_path = os.path.join(resultDir, G_model_filename)
        self.discriminator.load_state_dict(torch.load(D_model_path))
        self.generator.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))
    def to_np(self,x):
        return x.data.cpu().numpy()
    def real_images(self,images, number_of_images):
            return self.to_np(images.view(-1, 3, 256, 256)[:number_of_images])
    def generate_img(self,z, number_of_images):
        samples = self.generator(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
                generated_images.append(sample.reshape(3, 256, 256))
        return generated_images
    # =============================================================================
    #                       #Training GAN
    # =============================================================================
    def train(self,train_loader, results_dir,total_iterations = 100000,critic_iters = 5, save_training_gif=False,save_per_times = 30,log_per_times= 10, scan_splt =15):
        nz = 128 
        number_of_images = 10
        training_progress_images = []
        logger = Logger('{}/logs'.format(results_dir))
        start_time = time.time()
        d_load = get_infinite_batches(train_loader)
        for iteration in range(total_iterations):
            
            for p in self.discriminator.parameters():
                p.requires_grad=True
               #freeze generator
            
            real_scan = d_load.__next__()
            scan = real_scan.squeeze(0)
            
            # Splitting Scan into chunks of 15 (in order o train without getting a CUDA Out of Memory)
            for image in scan.split(scan_splt,0):
     
                batchSize = image.shape[0]
                
                #run discrimintnator k times
                d_losses = 0.0
                d_loss_real =0.0
                d_loss_fake = 0.0
                wasserstein_distance =0.00
                gradient_penalty = 0.0
                
                for d_iter in range(critic_iters):
    
                    
                    torch.cuda.empty_cache()
                    self.discriminator.zero_grad()
                    z = create_noise(batchSize, nz,self.device).detach()
                    real_data = Variable(image.to(self.device))
                    fake_data = self.generator(z)
                    
                    #train discriminator
                    training_d = self.train_discriminator(self.discriminator_optimizer, real_data, fake_data)
                    
                    torch.cuda.empty_cache()
                    d_losses=d_losses+training_d[0]
                    d_loss_real =d_loss_real+training_d[1].item()
                    d_loss_fake = d_loss_fake+training_d[2].item()
                    wasserstein_distance =wasserstein_distance + training_d[3].item()
                    gradient_penalty =gradient_penalty + training_d[4].item()
                print('Discriminator iteration: {}/{}, loss_fake: {}, loss_real: {}, D_loss:{}'.format(d_iter,critic_iters,d_loss_fake/critic_iters, d_loss_real/critic_iters, d_losses/critic_iters))
                    
                #Train generator
                for p in self.discriminator.parameters():
                    p.requires_grad=False
                # freeze discriminator
                torch.cuda.empty_cache()   
    
                self.generator.zero_grad()
                noise = create_noise(batchSize, nz,self.device)
                #train the generator network
                training_g = self.train_generator(self.generator_optimizer, noise)
                
                torch.cuda.empty_cache()
                print('Generator iteration: {}/{}, G_loss: {}'.format(iteration,total_iterations,training_g))
     
               
                """samples_list = []
                for i in range(scan_splt,scan.shape[0],scan_splt):
                    samples_list.append(generator(create_noise(scan_splt, nz)).cpu().detach())
                samples_list.append(generator(create_noise(scan.shape[0]-i, nz)).cpu().detach())
                samples = torch.cat(samples_list)"""
     
        
                if iteration % log_per_times == 0:
                    info = {'discriminator_loss': d_losses/critic_iters, 'real_dis_loss': d_loss_real/critic_iters, 
                            'fake_dis_loss': d_loss_fake/critic_iters,'generator_loss': training_g, 
                            'Wasserstein_Distance': wasserstein_distance/critic_iters, 'Gradient_Penalty': gradient_penalty/critic_iters,
                            'inception_score': 1}
                    for tag, value in info.items():                    
                         logger.scalar_summary(tag, value, iteration + 1)
    
                #Save models and images
    
                if iteration % save_per_times ==0:
                    
                    self.save_model(results_dir)
                    training_result_images_dir = os.path.join(results_dir, 'training_result_images')
                    if not os.path.exists(training_result_images_dir):
                        os.makedirs(training_result_images_dir)
                    noise_sample = create_noise(batchSize, nz, self.device)
                    samples = self.generator(noise_sample).cpu().detach()
            # =============================================================================
            #             #saving as numpy files
            # =============================================================================
                    #samples_numpy = samples.numpy()
                    #save_numpy(os.path.join(training_result_images_dir,'img_generatori_iter_{}.npy'.format(str(iteration).zfill(3))),samples_numpy)
                    #saving as grid of images
                    generated_samples = utils.make_grid(samples)
                    utils.save_image(generated_samples, os.path.join(training_result_images_dir, 'img_generatori_iter_{}.png'.format(str(iteration).zfill(3))))
        
                
            # =============================================================================
            #             #save GIFs
            # =============================================================================
                    if save_training_gif:
                        img_grid = np.transpose(generated_samples.numpy(), (1, 2, 0))
                        training_progress_images.append(img_grid)
                if save_training_gif and not training_progress_images:
                    imageio.mimsave(os.path.join(training_result_images_dir,'training_{}_epochs.gif'.format(total_iterations)),
                                        training_progress_images)
                    
        t_end = time.time()
        print('Time of training:{}'.format((t_end - start_time)))
        self.save_model(results_dir)


def train(train_loader,results_dir,total_iterations = 100000,critic_iters = 5, save_training_gif=False,save_per_times = 30,log_per_times= 10, scan_splt =15):
    # set the computation device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #=============================================================================
    # =============================================================================     
    #            # Initialize Generator and Descriminator 
    # =============================================================================
    
    generator = GoodGenerator().to(device)
    discriminator = Discriminator().to(device)
    
    discriminator.weight_init(mean=0.0, std=0.02)
    generator.weight_init(mean=0.0, std=0.02)
    
    # =============================================================================
    #            # Defining Parameters and Hyperparameters
    # =============================================================================
    learning_rate = 0.00005
    
    b1 = 0.5
    b2 = 0.999
    
    generator_iters = 10000

    
    # =============================================================================
    #                 ## Defining Optimizers
    # =============================================================================
    
    generator_optimizer= optim.Adam(generator.parameters(), lr=learning_rate, betas=(b1, b2))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(b1,b2))
    trainer = WGANTrainer(generator, discriminator, generator_optimizer, discriminator_optimizer, device)
    trainer.train(train_loader, results_dir,total_iterations,critic_iters, save_training_gif,save_per_times ,log_per_times, scan_splt)