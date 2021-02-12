# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:17:46 2020

@author: adds0
"""

import torch
import re
import time
import os

from torchvision import utils
from utils.utilities import *

from training.good_G_D import *
from training.wasserstein_GP_256 import *
#from utils.inceptionScore import *
from pathlib import Path
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from utils.data_load import make_data_loader
from utils.utilities import Plane,Diagnoses


def evaluate(train_loader, loading_dir, results_dir,no_of_iterations=3000,starting_case_id=0):
    nz = 128
    save_per_times = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_model('discriminator.pkl', 'generator.pkl', loading_dir)
    d_load = get_infinite_batches(train_loader)
    
    for i in range(no_of_iterations):
        real_scan = d_load.__next__()
        scan = real_scan.squeeze(0)
        
        #number of slices is set to 57 if scan has more than 57 slices to prevent a CUDA Out of Memory Exception
        batch_size = scan.shape[0] if scan.shape[0] <= 57 else 57
        
        testing_result_images_dir = os.path.join(results_dir, 'testing_result_images')
        if not os.path.exists(testing_result_images_dir):
            os.makedirs(testing_result_images_dir)
        
        z = create_noise(batch_size, nz, device)
        torch.cuda.empty_cache()
        samples = generator(z).cpu().detach()
        samples_numpy = samples.numpy()
        print(samples_numpy.shape)
        case_id = i+starting_case_id
        save_numpy(os.path.join(testing_result_images_dir,'{}.npy'.format(str(case_id).zfill(3))),samples_numpy)

        grid = utils.make_grid(samples)

        if i % save_per_times ==0:
            utils.save_image(grid, os.path.join(testing_result_images_dir, 'img_generatori_iter_{}.png'.format(str(i).zfill(3))))


data_path = Path('D:/OneDrive - University of Glasgow/Documnets/courses/Msc/Project/MRI/MRNet/resources')
train_path = data_path/'train'
valid_path = data_path/'valid'

starting_case_id=1130
# =============================================================================
#                           #ACL
# =============================================================================
loading_dir = 'D:/OneDrive - City, University of London/MSc Data Science City/Individual Project/GANKneeMRI2/src/model_weights/wgan_[1,1,0]'
plane = Plane.Axial.name
daignoses = Diagnoses.ACL
dataset_type = 'train'
axial_data_loader = make_data_loader(data_path, dataset_type, plane= plane, diagnoses_filter =daignoses )
results_dir = 'D:/eval/gan/{}/{}/{}/results_{}'.format( dataset_type,plane,str(daignoses),time.strftime("%Y%m%d-%H%M%S"))
test = evaluate(axial_data_loader, loading_dir, results_dir,starting_case_id=starting_case_id)
 #'augmented_acl_labels',daignoses 
 
# =============================================================================
#                       #meniscus
# =============================================================================
starting_case_id = starting_case_id + 3000

loading_dir = 'D:/OneDrive - City, University of London/MSc Data Science City/Individual Project/GANKneeMRI2/src/model_weights/wgan_[1,0,1]'
plane = Plane.Axial.name
daignoses = Diagnoses.Meniscus
dataset_type = 'train'
axial_data_loader = make_data_loader(data_path, dataset_type, plane= plane, diagnoses_filter =daignoses )
results_dir = 'D:/eval/gan/{}/{}/{}/results_{}'.format( dataset_type,plane,str(daignoses),time.strftime("%Y%m%d-%H%M%S"))
test = evaluate(axial_data_loader, loading_dir, results_dir,starting_case_id=starting_case_id )
#'augmented_meniscus_labels',daignoses 

# =============================================================================
#                           #None
# =============================================================================
starting_case_id = starting_case_id + 6000
loading_dir = 'D:/OneDrive - City, University of London/MSc Data Science City/Individual Project/GANKneeMRI2/src/model_weights/wgan_[0,0,0]'
plane = Plane.Axial.name
daignoses = Diagnoses.Normal
dataset_type = 'train'
axial_data_loader = make_data_loader(data_path, dataset_type, plane= plane, diagnoses_filter =daignoses )
results_dir = 'D:/eval/gan/{}/{}/{}/results_{}'.format( dataset_type,plane,str(daignoses),time.strftime("%Y%m%d-%H%M%S"))
test = evaluate(axial_data_loader, loading_dir, results_dir,starting_case_id=starting_case_id )


######CREATE LABELS FOR AUGMENTED DATASET
from glob import glob
def create_labels(dataset_dir,plane,labels_filename):
    with open('{}/{}.csv'.format(dataset_dir,labels_filename), mode='w', newline = '') as csv_file:
        fieldnames = ['case','abnormal','acl','meniscus']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for file in sorted(glob('{}/{}/**/**.npy'.format(dataset_dir,plane),recursive=True)):
            labels = re.sub(r"\[|\]","",file.split('\\')[1]).split(',')
            case_id = os.path.splitext(os.path.basename(file))[0]
            writer.writerow({'case':'{}'.format(str(case_id).zfill(3)),'abnormal':labels[0],'acl':labels[1],'meniscus':labels[2]})

dataset_dir = 'D:/eval/gan/train'
filename='augmented_labels'
plane = 'axial'
create_labels(dataset_dir,plane,filename)