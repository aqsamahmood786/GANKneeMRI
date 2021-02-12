@echo off

call conda update conda -y
call conda update --all -y
call conda env create -f environment.yml.yml
call conda activate project2020