#!/bin/bash
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --exclude=gpu1              # Exclude node gpu1
#SBATCH --partition=all             # Use the GPU partition
#SBATCH --job-name=Image_Processing # Job name
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --output=job_output_%j.log  # Redirect output to a file with job ID
#SBATCH --error=job_error_%j.log    # Redirect errors to a file with job ID
#SBATCH --no-requeue
#SBATCH --exclusive                 # Limit to one job per node

# Load environment (edit to match your cluster)
module load gnu9/9.4.0
module load cuda/10.2

# Change directory to repo if needed
# cd /path/to/your/repo

# Run step
python run_single_step.py "$1" "$2"
