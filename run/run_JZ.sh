#!/bin/bash
#SBATCH -J CASC
#SBATCH -N 1          # nodes number
#SBATCH -n 40          # CPUs number (on all nodes) 
##SBATCH -q qos_cpu-t3
##SBATCH --partition=cpu_dev
#SBATCH --partition=prepost
#SBATCH -o CASC.eo%j   #
#SBATCH -e CASC.eo%j   #
#SBATCH -t 01:59:00    # time limit
#SBATCH --export=NONE
#SBATCH -A whl@cpu # put here you account/projet name

# job information
cat << EOF
------------------------------------------------------------------
Job submit on $SLURM_SUBMIT_HOST by $SLURM_JOB_USER
JobID=$SLURM_JOBID Running_Node=$SLURM_NODELIST
Node=$SLURM_JOB_NUM_NODES Task=$SLURM_NTASKS
------------------------------------------------------------------
EOF

# Name of the file
file=$1
echo $file

path="/lustre/fswork/projects/rech/whl/rces071/Github/cascade/src/"
filepy="compute_spectrum_cascade.py"
export MONORUN="python"

module load miniforge/24.9.0

time ${MONORUN} ${path}'/'${filepy} ${file} 
