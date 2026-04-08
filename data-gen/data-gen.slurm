#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Mon Jan 08 2024 21:45:49 GMT+1100 (Australian Eastern Daylight Time)

# Partition for the job:
#SBATCH --partition=sapphire

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="sts-training"

# The project ID which this job should run under:
#SBATCH --account="punim2163"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# The amount of memory in megabytes per node:
#SBATCH --mem=8000

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-1:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load GCC/11.3.0
module load Rust/1.87.0

# The job command(s):
cargo run --release -- $@

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s