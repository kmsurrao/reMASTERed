#!/bin/bash                                                                                         
#SBATCH --qos=debug                                                                                 
#SBATCH --time=00:30:00                                                                             
#SBATCH --nodes=1                                                                                   
#SBATCH --constraint=haswell                                                                        

module load python
module load texlive

source activate my_mpi4py_env

python test.py
