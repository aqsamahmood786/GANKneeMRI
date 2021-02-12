@ECHO OFF
echo "Submitting AI Platform PyTorch job"

PARENT_DIR=$(cd ../ && pwd)
echo $PARENT_DIR

REM Creating Python Distribution
REM python ../setup.py sdist -d ../dist

python ./setup.py sdist 

REM The PyTorch image provided by AI Platform Training.
REM IMAGE_URI=gcr.io/cloud-ml-public/training/pytorch-cpu.1-4

set BUCKET_NAME=gs://d-gan-786-storage

REM TRAINER_PACKAGE_PATH=$PARENT_DIR/training
set TRAINER_PACKAGE_PATH="./trainer"

set MAIN_TRAINER_MODULE="./trainer.basenet_model_task"

set PACKAGE_STAGING_PATH=%BUCKET_NAME%

for /f "delims=" %%a in ('wmic OS Get localdatetime ^| find "."') do set DateTime=%%a

set Yr=%DateTime:~0,4%
set Mon=%DateTime:~4,2%
set Day=%DateTime:~6,2%
set Hr=%DateTime:~8,2%
set Min=%DateTime:~10,2%
set Sec=%DateTime:~12,2%

rem set SUBFILENAME=%Yr%-%Mon%-%Day%_(%Hr%-%Min%-%Sec%)
set SUBFILENAME=%Yr%%Mon%%Day%_%Hr%%Min%%Sec%

set JOB_NAME=basemrnet_model_%SUBFILENAME%

set JOB_DIR=%BUCKET_NAME%/jobs/%JOB_NAME%

REM REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
REM or use the default '`us-central1`'. The region is where the job will be run.
REM REGION = 'us-central1'
set REGION="us-east1"


set RUNTIME_VERSION=2.1
set PYTHON_VERSION=3.7
set PATH_TO_PACKAGED_TRAINER="./dist/knee_mri_package-0.1.tar.gz"

REM --master-image-uri=gcr.io/cloud-ml-public/training/pytorch-gpu.1-4 \
echo %JOB_DIR%
echo %MAIN_TRAINER_MODULE%
echo %PACKAGE_STAGING_PATH%
echo %PATH_TO_PACKAGED_TRAINER%
echo %REGION%
echo %PYTHON_VERSION%


gcloud ai-platform jobs submit training %JOB_NAME% 
          --module-name=%MAIN_TRAINER_MODULE% 
          --staging-bucket=%PACKAGE_STAGING_PATH% 
          --job-dir=%JOB_DIR% 
          --packages=%PATH_TO_PACKAGED_TRAINER% 
          --scale-tier=basic_gpu 
          --region=%REGION% 
          --python-version=%PYTHON_VERSION% 
          --runtime-version=%RUNTIME_VERSION% 
          -- 
          --out-file=basenet_model_results.pkl 
		  
# Stream the logs from the job
gcloud ai-platform jobs stream-logs %JOB_NAME%

PAUSE