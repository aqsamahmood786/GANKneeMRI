# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:29:25 2020

@author: adds0
"""

# import libraries

import numpy as np
import pandas as pd
import os
import torch

import torch.optim as optim

import torch.nn.functional as F
from utils.data_load import make_data_loader
from models.mrnet import MRNet
from utils.utilities import create_positiveRate_csv,create_losses_csv,load_weights,print_stats,save_losses,save_checkpoint,save_positiveRate
from utils.logger import Logger
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score
from numpy import savetxt
# calculate weights

class MRNetTrainer:
    
    def __init__(self,data_dir,out_dir, plane, dataset_type,augmented_dir = None, device = None):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.plane = plane
        self.dataset_type = dataset_type
        self.augmented_dir = augmented_dir
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            

        
        print(f'Creating models...')
        # Create a model for each diagnosis
        self.models = [MRNet().to(self.device), MRNet().to(self.device), MRNet().to(self.device)]
        
        print('Loading Weights if they exist...')
        #load previous weights if they exist
        load_weights(self.models,self.out_dir,self.plane)

        
    def calculate_weights(self):
        diagnoses = ['abnormal', 'acl', 'meniscus']
    
        labels_path = f'{self.data_dir}/{self.dataset_type}_labels.csv'
        labels_df = pd.read_csv(labels_path)
    
        weights = []
    
        for diagnosis in diagnoses:
            neg_count, pos_count = labels_df[diagnosis].value_counts().sort_index()
            weight = torch.tensor([neg_count / pos_count])
            weight = weight.to(self.device)
            weights.append(weight)
    
        return weights
    # optimization algorithm
    def make_adam_optimizer(self,model, lr, weight_decay):
        return optim.Adam(model.parameters(), lr, weight_decay = weight_decay)#torch.optim.SGD, Adam
    
    def make_lr_scheduler(self,optimizer,
                          #mode='min'
                          factor=0.3,#0.3
                          patience=5, threshold=0.0001,#5
                          verbose=True):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    
                                                    factor=factor,
                                                    patience=patience,
                                                    threshold =threshold,
                                                    verbose=verbose)
    
    def batch_forward_backprop(self,models, inputs, labels,  optimizers, data_dir, device):#criterions,
        losses = []
        preds = []
        
        
        for i, (model, label,  optimizer) in \
                enumerate(zip(models, labels[0], optimizers)):
            model.train()
            optimizer.zero_grad()
            out = model(inputs)
    
            pos_weights = self.calculate_weights() 
            for weight in pos_weights:
                loss = F.binary_cross_entropy_with_logits(out, label.unsqueeze(0), weight) 
            #loss = criterion(out, label.unsqueeze(0))#.unsqueeze(0)
            loss.backward()
            optimizer.step()
            
            output = torch.sigmoid(out)
            preds.append(output.item())
           
            losses.append(loss.item())
    
        return np.array(losses), np.array(preds)
    
    def batch_forward(self,models, inputs, labels, data_dir, device):
        preds = []
        losses = []
        
       
        for i, (model, label) in \
                enumerate(zip(models, labels[0])):
            model.eval()
            
            out = model(inputs)
            
            pos_weights = self.calculate_weights(data_dir, 'valid', device)
    
            for weight in pos_weights:
                loss = F.binary_cross_entropy_with_logits(out, label.unsqueeze(0), weight)
            #loss = criterion(out, label.unsqueeze(0))#.unsqueeze(0)
            losses.append(loss.item())
            
            output = torch.sigmoid(out)
            preds.append(output.item())
           
            
        return np.array(preds), np.array(losses)
    
    # Classification metrix 
    def get_accuracy(self,y_true, y_prob):
        y_true = np.array(y_true).transpose()
        y_prob =  np.array(y_prob).transpose()
        y_prob = np.where(y_prob <= 0.5, 0, y_prob)
        y_prob = np.where(y_prob > 0.5, 1, y_prob)
        accuracy =  [(lab==pred).sum().item() for lab, pred in zip(y_true,y_prob)]
    
        return accuracy
    
    def get_confusion_matrix(self,y_true, y_pred):
        
        y_true = np.array(y_true).transpose()
        y_pred =  np.array(y_pred).transpose()
        y_pred = np.where(y_pred <= 0.5, 0, y_pred)
        y_pred = np.where(y_pred > 0.5, 1, y_pred)
        confusionMatrix = [confusion_matrix(label, pred) for label,pred in zip(y_true, y_pred)]
    
        return confusionMatrix
    
    
    def get_recall(self,y_true, y_pred):
        y_true = np.array(y_true).transpose()
        y_pred =  np.array(y_pred).transpose()
        y_pred = np.where(y_pred <= 0.5, 0, y_pred)
        y_pred = np.where(y_pred > 0.5, 1, y_pred)
        recall = [recall_score(label, pred, average="weighted") for label,pred in zip(y_true, y_pred)]
        return recall
    def get_precision(self,y_true, y_pred):
        y_true = np.array(y_true).transpose()
        y_pred =  np.array(y_pred).transpose()
        y_pred = np.where(y_pred <= 0.5, 0, y_pred)
        y_pred = np.where(y_pred > 0.5, 1, y_pred)
        precision = [precision_score(label, pred, average="weighted") for label,pred in zip(y_true, y_pred)]
        return precision
    def get_fscore(self,y_true, y_pred):
        y_true = np.array(y_true).transpose()
        y_pred =  np.array(y_pred).transpose()
        y_pred = np.where(y_pred <= 0.5, 0, y_pred)
        y_pred = np.where(y_pred > 0.5, 1, y_pred)
        fscore = [f1_score(label, pred, average="weighted") for label,pred in zip(y_true, y_pred)]
    
        return fscore
    def update_lr_schedulers(self,lr_schedulers, batch_valid_losses):
        for scheduler, v_loss in zip(lr_schedulers, batch_valid_losses):
            scheduler.step(v_loss)
        
    
    def run(self, epochs, lr,batch_size, weight_decay):
        diagnoses_labels = ['abnormal', 'acl', 'meniscus']
        log_per_times= 1
        positiveRate_path = create_positiveRate_csv(self.out_dir,self.plane)
        losses_path = create_losses_csv(self.out_dir, self.plane)
    
        logger = Logger('{}/logs'.format(self.out_dir))
    
        print('Creating data loaders...')
    
        train_loader = make_data_loader(self.data_dir, 'train', self.plane, batch_size=batch_size,diagnoses_filter = None, augmented_dir = self.augmented_dir, device=self.device, shuffle=True)
        valid_loader = make_data_loader(self.data_dir, 'valid', self.plane, batch_size=batch_size,diagnoses_filter = None, augmented_dir = self.augmented_dir, device=self.device, shuffle=True)
    #'F:/eval/gan',
            
        # Calculate loss weights based on the prevalences in train set
    
        # pos_weights = calculate_weights(data_dir, 'train', device)
        # criterions = [F.binary_cross_entropy_with_logits(pos_weight=weight) for weight in pos_weights]
    
        optimizers = [self.make_adam_optimizer(model, lr, weight_decay) \
                      for model in self.models]
    
        lr_schedulers = [self.make_lr_scheduler(optimizer) for optimizer in optimizers]
    
        min_valid_losses = [np.inf, np.inf, np.inf]
    
        print(f'Training a model using {self.plane} series...')
        print(f'Checkpoints and losses will be save to {self.out_dir}')
    
        for epoch, _ in enumerate(range(epochs), 1):
            print(f'=== Epoch {epoch}/{epochs} ===')
          
            batch_train_losses = np.array([0.0, 0.0, 0.0])
            batch_valid_losses = np.array([0.0, 0.0, 0.0])
            
            train_preds = []
            train_labels = []
            
    
            for inputs, labels in train_loader:
                
                # move-tensors-to-GPU 
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_loss, batch_preds= self.batch_forward_backprop(self.models, inputs, labels, optimizers, self.data_dir, self.device)
                
    
                batch_train_losses += batch_loss
                train_labels.append(labels.detach().cpu().numpy().squeeze())
                train_preds.append(batch_preds)
            
            valid_preds = []
            valid_labels = []
            
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                batch_preds, batch_loss = \
                    self.batch_forward(self.models, inputs, labels, self.data_dir, self.device)
                batch_valid_losses += batch_loss
                
                valid_labels.append(labels.detach().cpu().numpy().squeeze())
                valid_preds.append(batch_preds)
                            
            batch_train_losses /= len(train_loader)
            batch_valid_losses /= len(valid_loader)
            
           #Accuracy
            train_accuracy = self.get_accuracy(train_labels, train_preds)
            trainAccuracy = [a/len(train_loader) for a in train_accuracy]
            valid_accuracy = self.get_accuracy(valid_labels, valid_preds)  
            validAccuracy = [b/len(valid_loader) for b in valid_accuracy]
            print('train_abnormal_accuracy: {:.3f}, train_acl_accuracy: {:.3f}, train_meniscus_accuracy:{:.3f}'.format(trainAccuracy[0], trainAccuracy[1], trainAccuracy[2]))
            print('valid_abnormal_accuracy:{:.3f}, valid_acl_acuuracy:{:.3f}, valid_meniscus_accuracy:{:.3f}'.format(validAccuracy[0], validAccuracy[1], validAccuracy[2]))
           
            # plot confusion matrix
            valid_cm = self.get_confusion_matrix(valid_labels, valid_preds)
            print(valid_cm[0]) # abnormal
            print(valid_cm[1])# acl
            print(valid_cm[2])# meniscus
            
            #roc_curve
            print(batch_train_losses)
            # calculate  f score, recall, precision
            fscore = self.get_fscore(valid_labels, valid_preds)
            precision = self.get_precision(valid_labels, valid_preds)
            recall = self.get_recall(valid_labels, valid_preds)
            
            print('abnormal_precision: {:.3f}, recall: {:.3f}, fscore: {:.3f}' .format(precision[0], recall[0], fscore[0]))
            print('acl_precision: {:.3f}, recall: {:.3f}, fscore: {:.3f}' .format(precision[1], recall[1], fscore[1]))
            print('meniscus_precision: {:.3f}, recall: {:.3f}, fscore: {:.3f}' .format(precision[2], recall[2], fscore[2]))
            
            print_stats(batch_train_losses, batch_valid_losses,
                        valid_labels, valid_preds)
            save_losses(batch_train_losses, batch_valid_losses, losses_path)
            
            save_positiveRate(valid_cm,positiveRate_path)
            self.update_lr_schedulers(lr_schedulers, batch_valid_losses)
    
            for i, (batch_v_loss, min_v_loss) in \
                    enumerate(zip(batch_valid_losses, min_valid_losses)):
    
                if batch_v_loss < min_v_loss:
                    save_checkpoint(epoch, self.plane, diagnoses_labels[i], self.models[i],
                                    optimizers[i], self.out_dir)
    
                    min_valid_losses[i] = batch_v_loss
            if epoch % log_per_times == 0:
                info = {'abnormal_train_loss': batch_train_losses[0], 'acl_train_loss': batch_train_losses[1],'meniscus_train_loss': batch_train_losses[2],
                        'abnormal_valid_loss': batch_valid_losses[0],'acl_valid_loss': batch_valid_losses[1],'meniscus_valid_loss': batch_valid_losses[2],
                        'abnormal_train_accuracy':trainAccuracy[0], 'acl_train_accuracy': trainAccuracy[1], 'meniscus_train_acuuracy': trainAccuracy[2],
                        'abnormal_valid_acuuracy': validAccuracy[0], 'acl_valid_accuracy': validAccuracy[1], 'menisucs_valid_acuuracy':validAccuracy[2]}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch + 1)
            savetxt(os.path.join(self.out_dir,'labels.csv'), valid_labels, delimiter=',')
            savetxt(os.path.join(self.out_dir,'preds.csv'), valid_preds, delimiter=',')
        #save_model(self.out_dir)
        # if("gs://" in self.out_dir):  
        #     save_to_bucket(losses_path,self.out_dir)
    
    #running baseline  model
    
    
