#!/bin/bash
echo "Submitting AI Platform PyTorch job"

PARENT_DIR=$(cd ../ && pwd)
echo $PARENT_DIR

# Creating Python Distribution
#python ../setup.py sdist -d ../dist

python ./setup.py sdist 

# The PyTorch image provided by AI Platform Training.
#IMAGE_URI=gcr.io/cloud-ml-public/training/pytorch-cpu.1-4


# BUCKET_NAME: Change to your bucket name.
BUCKET_NAME=gs://d-gan-786-storage

# TRAINER_PACKAGE_PATH=$PARENT_DIR/training
TRAINER_PACKAGE_PATH="./trainer"

MAIN_TRAINER_MODULE="./trainer.basenet_model_task"

PACKAGE_STAGING_PATH=$BUCKET_NAME

JOB_NAME=basemrnet_model_$(date +%Y%m%d_%H%M%S) # you need a new job name for every run

JOB_DIR=$BUCKET_NAME/jobs/$JOB_NAME

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the job will be run.
#REGION = 'us-central1'
REGION="us-east1"


RUNTIME_VERSION=2.1
PYTHON_VERSION=3.7
PATH_TO_PACKAGED_TRAINER="./dist/knee_mri_package-0.1.tar.gz"

#--master-image-uri=gcr.io/cloud-ml-public/training/pytorch-gpu.1-4 \
gcloud ai-platform jobs submit training ${JOB_NAME} \
          --module-name ${MAIN_TRAINER_MODULE} \
          --staging-bucket ${PACKAGE_STAGING_PATH} \
          --job-dir=${JOB_DIR} \
          --packages ${PATH_TO_PACKAGED_TRAINER} \
          --scale-tier basic_gpu \
          --region ${REGION} \
          --python-version ${PYTHON_VERSION} \
          --runtime-version ${RUNTIME_VERSION} \
          -- \
          --out-file=basenet_model_results.pkl \
		  
# Stream the logs from the job
gcloud ai-platform jobs stream-logs ${JOB_NAME}

# Verify the model was exported
#echo "Verify the model was exported:"
#gsutil ls ${JOB_DIR}/basemrnet_model_*