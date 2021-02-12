# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 21:01:41 2020

@author: adds0
"""

# =============================================================================
# ## Import libraries
# =============================================================================
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import time
from training.baseline_model_training import MRNetTrainer
from utils.utilities import Plane
import argparse
import sys
from pathlib import Path
def parse_arguments(argv):
    print(argv)
    parser = argparse.ArgumentParser() # get a parser object
    parser.add_argument('--base-results-dir', metavar='base_results_dir', required=True,
                        help="The base directory for the results. The results would be stored '<base-dir>/results/gan/<dataset-type>/<plane>/<diagnoses>/results_<timestamp>' ") # add a required argument
    parser.add_argument('--data-dir', metavar='data_dir', required=True,
                        help='The Directory which contain the Original dataset.') # add a required argument
    parser.add_argument('--augmented-data-dir', metavar='augmented_data_dir', required=False,
                    help='The Directory which contain the Augmented dataset.') # add a required argument
    parser.add_argument('--plane', metavar='plane', required=True,choices=[e.name.lower() for e in Plane],
                        help='The plane of MRI Directory.') # add a required argument
    parser.add_argument('--dataset-type', metavar='dataset_type', required=True,choices=['train','test'])
    parser.add_argument('--learning-rate', metavar='learning_rate', required=False,default=0.00001,
                  help='Model Learning rate.')    
    parser.add_argument('--epochs', metavar='epochs', required=False,default=100,
                  help='The number of epochs to train model.') # add a required argument    
    parser.add_argument('--batch-size', metavar='batch_size', required=False,default=1,
                  help='The number of scan batches')     
    parser.add_argument('--weight-decay', metavar='weight_decay', required=False,default=0.01,
                  help='Weight Decay')                     
    return  parser.parse_args(argv)

"""
#C:/Users/adds0/
data_path = "C:/Users/adds0/OneDrive - University of Glasgow/Documnets/courses/Msc/Project/MRI/MRNet/resources"
#data_path = "F:/dataset"
plane = Plane.Axial.name
dataset_type = 'train'
results_dir = 'D:/MRNET/{}/{}/results_{}'.format( dataset_type,plane,time.strftime("%Y%m%d-%H%M%S"))
#results_dir = 'D:/MRNET/{}/{}/results_20201102-114510'.format( dataset_type,plane)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
mrnet_trainer = MRNetTrainer(data_path,results_dir,plane,dataset_type,None,None)
mrnet_trainer.run( epochs = 100, lr = 0.00001,batch_size =1, weight_decay = 0.01) #0.01,0.001, 0.0001# weight decay, 0.03,0.01 
#axial_data_loader = make_data_loader(data_path, 'train', 'axial',diagnoses = None,augmented_dir = 'F:/eval/gan', device = None)
#0.001,0.001
#Prameters for baseline model
#learning rate = 0.00001, 0.01, patience = 1, factor, 0.3, adam
#other mode 
#learning rate = 0.001, 0.001, patience = 5, factor, 0.1, SGD
#3rd model
#SGD, lr 0.0001, weight_decay = 0.03
#4th model (geometric and wgan images)
#Adam, 0.0001, 0.0001
"""
def main():
    argv = sys.argv[1:]
    args = parse_arguments(argv)
    data_path = Path(args.data_dir)
    train_path = data_path/'train'
    valid_path = data_path/'valid'
    results_dir_template = '{}/gan/{}/{}/{}/results_{}'
    
    plane = Plane(args.plane)
    dataset_type = args.dataset_type
    
    mrnet_trainer = MRNetTrainer(args.data_dir,args.base_results_dir,args.plane,args.dataset_type,None,args.augmented_data_dir)
    mrnet_trainer.run( epochs = int(args.epochs), lr = args.learning_rate,batch_size =args.batch_size, weight_decay = args.weight_decay)
    
if __name__ == "__main__":
    main()