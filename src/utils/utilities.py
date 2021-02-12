# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 00:55:31 2020

@author: adds0
"""
import os
import csv
import numpy as np
from sklearn import metrics
import pickle
import torch
from glob import glob
import re
import enum 
from torch.autograd import Variable

MAX_PIXEL_VAL = 255
MEAN = 58.09
STD = 49.73

def create_losses_csv(out_dir, plane):
    losses_path = f'{out_dir}/losses_{plane}.csv'

    with open(f'{losses_path}', mode='w') as losses_csv:
        fields = ['t_abnormal', 't_acl', 't_meniscus',
                    'v_abnormal', 'v_acl', 'v_meniscus']
        writer = csv.DictWriter(losses_csv, fieldnames=fields)
        writer.writeheader()

    return losses_path


def create_positiveRate_csv(out_dir, plane):
    positiveRate_path = f'{out_dir}/positiveRate_{plane}.csv'

    with open(f'{positiveRate_path}', mode='w') as positiveRate_csv:
        fields = ['v_abnormal_PR', 'v_acl_PR', 'v_meniscus_PR']
        writer = csv.DictWriter(positiveRate_csv, fieldnames=fields)
        writer.writeheader()

    return positiveRate_path
def save_positiveRate(confusionmatrix, positiveRate_path):
    with open(f'{positiveRate_path}', mode='a') as positiveRate_csv:
        writer = csv.writer(positiveRate_csv)
        writer.writerow(writer.writerow(r) for r in confusionmatrix)
def create_labels_csv(losses_path ):

    with open(f'{losses_path}', mode='w') as losses_csv:
        fields = ['case','abnormal', 'acl', 'meniscus']
        writer = csv.DictWriter(losses_csv, fieldnames=fields)
        writer.writeheader()

    return losses_path

def save_numpy(numpy_filepath,numpy_array ):

    with open(f'{numpy_filepath}', mode='wb') as f:
        np.save(f, numpy_array)

def create_output_dir(job_dir,exp, plane):
    out_dir = f'./models/{exp}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    losses_path = create_losses_csv(out_dir, plane)

    return out_dir, losses_path
def calculate_aucs(all_labels, all_preds):
    all_labels = np.array(all_labels).transpose()
    all_preds =  np.array(all_preds).transpose()

    aucs = [metrics.roc_auc_score(labels, preds) for \
            labels, preds in zip(all_labels, all_preds)]

    return aucs
def print_stats(batch_train_losses, batch_valid_losses,
                valid_labels, valid_preds):
    aucs = calculate_aucs(valid_labels, valid_preds)
    
    print(f'Train losses - abnormal: {batch_train_losses[0]:.3f},',
          f'acl: {batch_train_losses[1]:.3f},',
          f'meniscus: {batch_train_losses[2]:.3f}',
          f'\nValid losses - abnormal: {batch_valid_losses[0]:.3f},',
          f'acl: {batch_valid_losses[1]:.3f},',
          f'meniscus: {batch_valid_losses[2]:.3f}',
          f'\nValid AUCs - abnormal: {aucs[0]:.3f},',
          f'acl: {aucs[1]:.3f},',
          f'meniscus: {aucs[2]:.3f}')
def save_losses(train_losses , valid_losses, losses_path):
    with open(f'{losses_path}', mode='a') as losses_csv:
        writer = csv.writer(losses_csv)
        writer.writerow(np.append(train_losses, valid_losses))
def save_to_bucket(filename,bucket):
    import subprocess
    proc = subprocess.run(["gsutil","cp", filename, bucket],stderr=subprocess.PIPE)
    print("gstuil returned: " + str(proc.returncode))
    print(str(proc.stderr))

def save(object, bucket, filename):
    with open(filename,mode='wb') as f:
        pickle.dump(object,f)
        
    print("Saving {} to {}".format(filename,bucket))
    save_to_bucket(filename,bucket)
    
def save_checkpoint(epoch, plane, diagnosis, model, optimizer, out_dir):
    print(f'Min valid loss for {diagnosis}, saving the checkpoint...')

    checkpoint = {
        'epoch': epoch,
        'plane': plane,
        'diagnosis': diagnosis,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    chkpt = f'cnn_{plane}_{diagnosis}_{epoch:02d}.pt'
    if("gs://" in out_dir):
        save(checkpoint,out_dir,chkpt)
    else:
        torch.save(checkpoint, f'{out_dir}/{chkpt}')


def load_weights(models,out_dir,plane):
    checkpointFileNames = glob('{}/cnn_{}_*.pt'.format(out_dir,plane))
    if(not checkpointFileNames):
        print("No Weights Loaded!")
        return
    latest_checkpoint_files = [file for file in checkpointFileNames if max([re.match(r'.*(\d+)\.pt', i).group(1) for i in checkpointFileNames if re.match(r'.*(\d+)\.pt', i)]) in file]
    for i in range(len( latest_checkpoint_files)):
        checkpoint = torch.load(latest_checkpoint_files[i])
        models[i].load_state_dict(checkpoint['state_dict'])
        print("Loading Weights for {}".format(latest_checkpoint_files[i]))
        
def free_cuda_memory(obj):
    del obj
    torch.cuda.empty_cache()
    
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.npy'])

# Helping Functions

def label_real(size,device):
    """
    Fucntion to create real labels (ones)
    :param size: batch size
    :return real label vector
    """
    data = Variable(torch.ones((size, 1)).to(device))

    return data
def label_fake(size,device):
    """
    Fucntion to create fake labels (zeros)
    :param size: batch size
    :returns fake label vector
    """
    data = Variable(torch.zeros((size, 1)).to(device))
    return data

def create_noise(sample_size, nz,device):
    """
    Fucntion to create noise
    :param sample_size: fixed sample size or batch size
    :param nz: latent vector size
    :returns random noise vector
    """
    data =Variable(torch.randn(sample_size, nz)).normal_(0, 1).to(device)
    
    return data.to(device)

def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images
       
class Plane(enum.Enum):
    def __str__(self):
        return str(self.value)
    Axial = 'axial'
    Sagittal = 'sagittal'
    Coronal = 'coronal' 
    

class Diagnoses(enum.Enum):  
    def __str__(self):
        return str(self.value)
    Abnormal = ([1,0,0])
    ACL = ([1,1,0])
    Meniscus = ([1,0,1])
    All = ([1,1,1])
    Normal = ([0,0,0]) 
    
    def __init__(self, list_encoded):
        self.list_encoded = list_encoded   
   
    @property
    def encoded(self):
        return self.list_encoded
    
