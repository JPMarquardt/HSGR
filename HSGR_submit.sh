#!/bin/bash


#SBATCH --job-name=nfflr_train
#SBATCH --output=logs/nfflr-%j.out
#SBATCH --error=logs/nfflr-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu

module purge

singularity exec --nv \
	    --overlay /scratch/jpm9731/HSGR-singularity/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; nvidia-smi; python HSGR_nfflr.py"
