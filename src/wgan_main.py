# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 00:19:52 2020

@author: adds0
"""
# =============================================================================
# ## Import libraries
# =============================================================================
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
from utils.data_load import make_data_loader
from pathlib import Path
from training.wasserstein_GP_256 import train
from torchvision import datasets
from utils.utilities import Plane,Diagnoses
import argparse
import sys
def parse_arguments(argv):
    print(argv)
    parser = argparse.ArgumentParser() # get a parser object
    parser.add_argument('--base-results-dir', metavar='base_results_dir', required=True,
                        help="The base directory for the results. The results would be stored '<base-dir>/results/gan/<dataset-type>/<plane>/<diagnoses>/results_<timestamp>' ") # add a required argument
    parser.add_argument('--data-dir', metavar='data_dir', required=True,
                        help='The Directory which contain the Original dataset.') # add a required argument
    parser.add_argument('--plane', metavar='plane', required=True,choices=[e.name.lower() for e in Plane],
                        help='The plane of MRI Directory.') # add a required argument
    parser.add_argument('--diagnoses', metavar='diagnoses', required=True,choices=[d.name for d in Diagnoses],
                        help='Filter dataset according to the Diagnoses.') # add a required argument
    parser.add_argument('--dataset-type', metavar='dataset_type', required=True,choices=['train','test'],
                        help='The Datset Type.') # add a required argument
    parser.add_argument('--total-iterations', metavar='total_iterations', required=False,default=1000000,
                  help='The number of iterations.') # add a required argument                      
    return  parser.parse_args(argv)

def create_results_dir(directory_template,base_dir,dataset_type,plane,daignoses,time):
    return directory_template.format( base_dir,dataset_type,plane,str(daignoses).strip('[],').replace(',','_'),time)
"""
############################################################################################
data_path = Path('C:/Users/adds0/OneDrive - University of Glasgow/Documnets/courses/Msc/Project/MRI/MRNet/resources')
train_path = data_path/'train'
valid_path = data_path/'valid'
results_dir_template = 'D:/results/gan/{}/{}/{}/results_{}'

# =============================================================================
# ### Axial_Abnormal
# =============================================================================
plane = Plane.Axial.name

daignoses = Diagnoses.Abnormal
dataset_type = 'train'
axial_data_loader = make_data_loader(data_path, dataset_type, plane= plane, diagnoses_filter =daignoses )

results_dir = create_results_dir(results_dir_template,dataset_type,plane,daignoses,time.strftime("%Y%m%d-%H%M%S"))

train_gan = train(axial_data_loader, results_dir,total_iterations=1000000)
# =============================================================================
# ## Axial Meniscus
# =============================================================================
plane = Plane.Axial.name
daignoses =  Diagnoses.Meniscus
dataset_type = 'train'
axial_data_loader = make_data_loader(data_path, dataset_type, plane= plane, diagnoses_filter =daignoses )
results_dir = create_results_dir(results_dir_template,dataset_type,plane,daignoses,time.strftime("%Y%m%d-%H%M%S"))
train_gan = train(axial_data_loader, results_dir,total_iterations=1000000)
# =============================================================================
# ## Axial ACL
# =============================================================================
plane = Plane.Axial.name
daignoses =  Diagnoses.ACL
dataset_type = 'train'
axial_data_loader = make_data_loader(data_path, dataset_type, plane= plane, diagnoses_filter =daignoses )

results_dir = create_results_dir(results_dir_template,dataset_type,plane,daignoses,time.strftime("%Y%m%d-%H%M%S"))
train_gan = train(axial_data_loader, results_dir,total_iterations=100000)

# =============================================================================
# ## Axial ACL Abonormal Meniscus
# =============================================================================
plane = Plane.Axial.name
daignoses = Diagnoses.All
dataset_type = 'train'
axial_data_loader = make_data_loader(data_path, dataset_type, plane= plane, diagnoses_filter =daignoses )

results_dir = create_results_dir(results_dir_template,dataset_type,plane,daignoses,time.strftime("%Y%m%d-%H%M%S"))
train_gan = train(axial_data_loader, results_dir,total_iterations=100000)

# =============================================================================
# ## Axial None
# =============================================================================
plane = Plane.Axial.name
daignoses =  Diagnoses.Normal
dataset_type = 'train'
axial_data_loader = make_data_loader(data_path, dataset_type, plane= plane, diagnoses_filter =daignoses )

results_dir = create_results_dir(results_dir_template,dataset_type,plane,daignoses,time.strftime("%Y%m%d-%H%M%S"))
train_gan = train(axial_data_loader, results_dir,total_iterations=100000)

"""
def main():
    argv = sys.argv[1:]
    args = parse_arguments(argv)
    data_path = Path(args.data_dir)
    train_path = data_path/'train'
    valid_path = data_path/'valid'
    results_dir_template = '{}/gan/{}/{}/{}/results_{}'
    
    plane = Plane(args.plane)
    daignoses = Diagnoses[args.diagnoses]
    dataset_type = args.dataset_type
    
    axial_data_loader = make_data_loader(data_path, dataset_type, plane= plane, diagnoses_filter =daignoses )
    
    results_dir = create_results_dir(results_dir_template,args.base_results_dir,dataset_type,plane,daignoses,time.strftime("%Y%m%d-%H%M%S"))
    train_gan = train(axial_data_loader, results_dir,total_iterations=args.total_iterations)
    
if __name__ == "__main__":
    main()
    