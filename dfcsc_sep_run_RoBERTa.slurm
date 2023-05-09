#!/bin/bash

#SBATCH --job-name=DFCSC-SEP_RoBERTa    # Job name
#SBATCH --time=0-7:0    # Sets the maximum excution time of the job. The pattern is days-hours:minutes

#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1

#SBATCH --mail-type=ALL                        # All messages will be notified by email
#SBATCH --mail-user=alexlimatds@yahoo.com.br   # E-mail address to receive notifications

##SBATCH --partition=test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Job
module load singularity/3.7.1
singularity exec --nv /opt/npad/shared/containers/pytorch_22.07-py3.sif python dfcsc_sep_run_RoBERTa.py
