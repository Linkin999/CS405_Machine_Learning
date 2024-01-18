#!/bin/bash
#SBATCH -o job.%j.out          
#SBATCH --partition=gpulab02
#SBATCH --qos=gpulab02
#SBATCH -J myFirstGPUJob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --job-name=Unet

nvidia-smi

python3 test_cuda.py