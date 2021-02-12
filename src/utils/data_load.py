#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import os
import torch
import torch.nn.parallel
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from utils.utilities import *
from torch.utils.data.dataset import Dataset
import pandas as pd
from torchvision import transforms
from glob import glob
import cv2

MAX_PIXEL_VAL = 255
MEAN = 58.09
STD = 49.73


def preprocess_data(case_path,case_id, transform=None):
    series =np.load(case_path).astype(np.float32)
 
    if len(series.shape) == 3:
       series = torch.tensor(np.stack((series,)*3, axis=1))
    else:
        series = torch.tensor(series)
    transformed_series = []
    if transform is not None:
        for i, slice in enumerate(series.split(1)):
            #if case_id < 1130:
            transformed_series.append( transform(slice.squeeze()))
            #else:
                ##do not apply transformations on augmented images
                #transformed_series.append( slice.squeeze())
    #   # Normalization
    series = torch.stack(transformed_series)
    
    #series = (series - series.min()) / (series.max() - series.min()) * MAX_PIXEL_VAL
    #series = (series - MEAN) / STD
    
    return series
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MRNetDataset(Dataset):
    def __init__(self, dataset_dir, labels_path,plane,diagnoses_filter=None, transform=None,augmented_dir=None, device=None):

        self.case_paths = sorted(glob('{}/{}/**.npy'.format(dataset_dir,plane)))


        self.labels_df = pd.read_csv(labels_path)
        
        if augmented_dir is not None:

            self.case_paths.extend( sorted(glob('{}/{}/**/**.npy'.format(augmented_dir,plane),recursive=True)))
            self.labels_df = self.labels_df.append(pd.read_csv('{}/{}_labels.csv'.format(augmented_dir,'augmented')), ignore_index=True)

        #self.resize = resize
        self.diagnoses_filter = diagnoses_filter

        if diagnoses_filter is not None:
            encoded_diagnoses = diagnoses_filter.encoded
            self.labels_df = self.labels_df[(self.labels_df['abnormal'] == encoded_diagnoses[0]) & (self.labels_df['acl'] == encoded_diagnoses[1]) & (self.labels_df['meniscus'] == encoded_diagnoses[2]) ]
            filtered_files = list(filter(lambda x: int(os.path.splitext(os.path.basename(x))[0]) in list(self.labels_df.case), self.case_paths))
            self.case_paths = filtered_files

        self.transform = transform
        #self.window = 10
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.case_paths)

    def __getitem__(self, idx):
        case_path = self.case_paths[idx]
        case_id = int(os.path.splitext(os.path.basename(case_path))[0])
        

        series = preprocess_data(case_path,case_id, self.transform)

        case_row = self.labels_df[self.labels_df.case == case_id]
        diagnoses = case_row.values[0,1:].astype(np.float32)
        labels = torch.tensor(diagnoses)


        return (series, labels)

def make_dataset(data_dir, dataset_type, plane,diagnoses_filter=None,augmented_dir=None, device=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'



    dataset_dir = '{}/{}'.format(data_dir, dataset_type)
    labels_path = '{}/{}_labels.csv'.format(data_dir,dataset_type)

    if dataset_type == 'train':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(p =0.5),
            #transforms.RandomRotation((-20, 20)),
            
            #transforms.GaussianBlur((0.1,0.2)),
            #RandomGaussBlur(radius=[0.1, 0.2]),
            transforms.RandomAffine(20,translate=(0.1, 0.1)),#scale=(1.3, 1.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0, hue=0),
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomAffine(25, translate=(0.1, 0.1)),
            
            transforms.ToTensor(),
            #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    elif dataset_type == 'valid':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    else:
        raise ValueError('Dataset needs to be train or valid.')
    augmented_directory = '{}/{}'.format(augmented_dir, dataset_type) if augmented_dir is not None else None
    dataset = MRNetDataset(dataset_dir, labels_path, plane,diagnoses_filter,transform = transform,augmented_dir=augmented_directory, device=device)
    
    return dataset


import torch
from torch.utils.data import DataLoader

#from dataset import make_dataset


def make_data_loader(data_dir, dataset_type, plane,batch_size = 1,diagnoses_filter = None,augmented_dir=None, device=None, shuffle=True):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = make_dataset(data_dir, dataset_type, plane,diagnoses_filter, augmented_dir, device=device)
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)

    return data_loader

def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images





