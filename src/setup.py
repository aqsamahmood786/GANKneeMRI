# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 23:00:41 2020

@author: adds0
"""

from setuptools import setup
from setuptools import find_packages
#'torchvision==0.5.0', 'torch @ https://download.pytorch.org/whl/cu102/torch-1.5.1-cp36-cp36m-linux_x86_64.whl',' cudatoolkit==10.2'
REQUIRED_PACKAGES = ['torch @ https://download.pytorch.org/whl/cu101/torch-1.4.0-cp37-cp37m-linux_x86_64.whl',
                     'torchvision @ https://download.pytorch.org/whl/cu101/torchvision-0.5.0-cp37-cp37m-linux_x86_64.whl',
                     'opencv-python', 'facenet-pytorch','matplotlib ','scikit-image','protobuf','gcsfs']

setup(
 name="knee_mri_package",
 version="0.1",
 include_package_data=True,
 #scripts=["data_load.py", "logger.py", "axial_2D_WGAN.py", "axial_WGAN_training"],
 install_requires=REQUIRED_PACKAGES,
 packages=find_packages()
)