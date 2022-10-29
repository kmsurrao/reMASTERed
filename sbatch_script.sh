#!/bin/bash
# Anaconda Environment submit script for Slurm.
#
#SBATCH --account=hill # The account name for the job.
#SBATCH --job-name=MyCondaEnv # The job name.
#SBATCH -c 8 # The number of cpu cores to use.
#SBATCH --time=0-3:30 # The time the job will take to run.
#SBATCH --mem-per-cpu=20gb # The memory the job will use per cpu core.


# Load anaconda & gcc
module load anaconda/3-5.3.1
module load gcc/7.2.0

# Activate environment
source activate /moto/hill/users/kms2320/MyEnv

# Command to execute Python program
/moto/hill/users/kms2320/MyEnv/bin/python consistency_checks.py moto.yaml
# end of script
