#!/bin/bash
#!/scratch/jpm9731/.conda/envs/bin/python3.11
#SBATCH --job-name=nfflr_train
#SBATCH --output=outs/nfflr-%j.out
#SBATCH --error=outs/nfflr-%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB

module purge
module load anaconda3/2020.07
source activate /home/jpm9731/.conda/envs/HSGR
python --version

python /scratch/jpm9731/HSGR/HSGR_nfflr.py
