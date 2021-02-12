# IMPORT LIBRARIES
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils.data_load import make_data_loader
from training.wasserstein_GP_256 import import train
from utils.utilities import Plane,Diagnoses
import datetime
import time
import argparse
import pickle
import sys
PROJECT = 'd-gan-786' 
BUCKET = 'gs://{}-storage'.format(PROJECT)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def parse_arguments(argv):
    print(argv)
    parser = argparse.ArgumentParser() # get a parser object
    parser.add_argument('--out-file', metavar='out_file', required=True,
                        help='The filename for the result.') # add a required argument
    parser.add_argument('--data-dir', metavar='data_dir', required=True,
                        help='The Job Directory.') # add a required argument
    parser.add_argument('--plane', metavar='plane', required=True,choices=[e.name for e in Plane],
                        help='The plane of MRI Directory.') # add a required argument
    parser.add_argument('--diagnoses', metavar='diagnoses', required=True,required=True,choices=[e.name for e in Diagnoses],
                        help='Filter dataset according to the Diagnoses.') # add a required argument
    parser.add_argument('--dataset_type', metavar='dataset_type', required=True,
                        help='The Datset Type.') # add a required argument
    parser.add_argument('--worker-count', metavar='worker_count', required=False,default=1,
                  help='The number of worker nodes.') # add a required argument                      
    return  parser.parse_args(argv)

if  'google.colab' not in sys.modules: # Don't use system arguments when run in Colab 
    # Parse the provided arguments
    argv = sys.argv[1:]
    args = parse_arguments(argv)
    data_dir = args.data_dir
    start_time = time.time()
    gcs_pattern = 'gs://d-gan-786-storage'
    #filenames = tf.io.gfile.glob(gcs_pattern)
    plane = args.plane
    diagnoses = Diagnoses[args.diagnoses]
    dataset_type = args.dataset_type
    axial_data_loader = make_data_loader(data_dir, dataset_type=dataset_type, plane= plane, diagnoses_filter =diagnoses )
    
    results_dir = '{}/results/gan/{}/{}/{}/results_{}'.format( BUCKET,dataset_type,plane,diagnoses,time.strftime("%Y%m%d-%H%M%S"))
    
    #axial_data_test_loader = make_data_loader(data_path, test, plane= plane, diagnoses =daignoses )
    train_gan = train(axial_data_loader, results_dir,total_iterations=100000) 
    print("time taken to run model {}".format(time.time()-start_time))
elif __name__ == "__main__" : # but define them manually
    JOB_NAME = "WGAN_training_{}".format(datetime.datetime.now().strftime("%y%m%d-%H%M"))
    DATA_DIR = BUCKET + '/resources'
    argv = [ "--out-file", "task4_results.pkl","--data-dir", DATA_DIR,"--plane",'axial',"--dataset_type",'train',"--diagnoses",'acl']
    args = parse_arguments(argv)
    dataset_type = args.dataset_type
    plane= args.plane
    diagnoses = Diagnoses(args.diagnoses)

    start_time = time.time()
    gcs_pattern = 'gs://d-gan-786-storage'
    #filenames = tf.io.gfile.glob(gcs_pattern)
    axial_data_loader = make_data_loader(args.data_dir, dataset_type=args.dataset_type, plane= args.plane, diagnoses_filter =args.diagnoses )
    results_dir = '{}/results/gan/{}/{}/{}/results_{}'.format( BUCKET,dataset_type,plane,diagnoses,time.strftime("%Y%m%d-%H%M%S"))
    #axial_data_test_loader = make_data_loader(data_path, test, plane= plane, diagnoses =daignoses )
    train_gan = train(axial_data_loader, results_dir,total_iterations=100000)
    print("time taken to run model {}".format(time.time()-start_time))