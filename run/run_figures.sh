#!/bin/bash
#SBATCH -J FIG
#SBATCH -N 1          # nodes number
#SBATCH -n 1          # CPUs number (on all nodes) 
##SBATCH -q qos_cpu-t3
##SBATCH --partition=cpu_p1
#SBATCH --partition=prepost
#SBATCH -o FIG.eo%j   #
#SBATCH -e FIG.eo%j   #
#SBATCH -t 04:59:00    # time limit
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
#file=$1
#echo $file

path="/lustre/fswork/projects/rech/whl/rces071/Github/cascades/src/"
filepy="plot_spectrum_cascade.py"
export MONORUN="python"

module load miniforge/24.9.0

time ${MONORUN} ${path}'/'${filepy} #${file} 
