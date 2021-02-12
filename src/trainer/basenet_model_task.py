# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 20:29:55 2020

@author: adds0
"""
# IMPORT LIBRARIES
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sys
from training.baseline_model_training import MRNetTrainer
from utils.utilities import Plane
import datetime
import time
import argparse


PROJECT = 'd-gan-786' 
BUCKET = 'gs://{}-storage'.format(PROJECT)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def parse_arguments(argv):
    print(argv)
    parser = argparse.ArgumentParser() # get a parser object
    parser.add_argument('--out-file', metavar='out_file', required=False,
                        help='The filename for the result.') # add a required argument
    parser.add_argument('--job-dir', metavar='job_dir', required=False,
                        help='The Job Directory.') # add a required argument
    parser.add_argument('--worker-count', metavar='worker_count', required=False,default=1,
                  help='The number of worker nodes.') # add a required argument   
    parser.add_argument('--plane', metavar='plane', required=True,default='axial',
                  help='mri plane.') # add a required argument  
    parser.add_argument('--dataset-type', metavar='dataset_type', required=True,default='train',
              help='Train or Test dataset.') # add a required argument
    parser.add_argument('--learning-rate', metavar='lr', required=False,default=0.00001,
                  help='Model Learning rate.') #    
    parser.add_argument('--epochs', metavar='ep', required=False,default=100,
                  help='The number of epochs to train model.') # add a required argument    
    parser.add_argument('--batch-size', metavar='batch_size', required=False,default=1,
                  help='The number of scan batches')     
    parser.add_argument('--weight-decay', metavar='weight_decay', required=False,default=0.01,
                  help='Weight Decay')                  
    return  parser.parse_args(argv)

if  'google.colab' not in sys.modules: # Don't use system arguments when run in Colab 
    # Parse the provided arguments
    argv = sys.argv[1:]
    args = parse_arguments(argv)
  
    start_time = time.time()
    gcs_pattern = 'gs://d-gan-786-storage'
    mrnet_trainer = MRNetTrainer('{}/resources'.format(BUCKET),args.job_dir,args.plane,args.dataset_type,None,None)
    mrnet_trainer.run( epochs = args.ep, lr = args.lr,batch_size =args.batch_size, weight_decay = args.weight_decay)
    
    print("time taken to run model {}".format(time.time()-start_time))
elif __name__ == "__main__" : # but define them manually
    JOB_NAME = "basemrnet_model{}".format(datetime.datetime.now().strftime("%y%m%d-%H%M"))
    JOB_DIR = BUCKET + '/jobs/' + JOB_NAME
    argv = [ "--out-file", "basenet_model_results.pkl","--job-dir", JOB_DIR,"--plane","axial","--dataset-type","train","--epochs",50,"--learning-rate",0.00001,"--batch-size",1]
    args = parse_arguments(argv)
    start_time = time.time()
    gcs_pattern = 'gs://d-gan-786-storage'

    mrnet_trainer = MRNetTrainer('{}/resources'.format(BUCKET),args.job_dir,args.plane,args.dataset_type,None,None)
    mrnet_trainer.run( epochs = args.ep, lr = args.lr,batch_size =args.batch_size, weight_decay = args.weight_decay)
    
    print("time taken to run model {}".format(time.time()-start_time))