#!/bin/bash
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --exclude=gpu1              # Exclude node gpu1
#SBATCH --partition=all             # Use the GPU partition
#SBATCH --job-name=Image_Processing # Job name
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --output=pipelines/job_output_%j.log  # Redirect output to pipelines directory with job ID
#SBATCH --error=pipelines/job_error_%j.log    # Redirect errors to pipelines directory with job ID
#SBATCH --no-requeue
#SBATCH --exclusive                 # Limit to one job per node

# Load environment (edit to match your cluster)
module load gnu9/9.4.0
module load cuda/10.2

echo "Starting my job..."
start_time=$(date +%s)

# Ensure output directory exists
# mkdir -p "${PWD}/output"

# # Correct output file name
# output_names="${PWD}/output/output_${SLURM_JOB_ID}_$(basename ${fileLocation})"

# Activate the environment and run the script
source ../.venv/bin/activate
export QT_QPA_PLATFORM=offscreen

# Run step
python run_pipeline.py "$1" &

wait

# End timing the process
end_time=$(date +%s)
total_time=$(((end_time - start_time) / 60 ))

# Print the time to complete the process
echo "Total time to complete the job: $total_time minutes"

exit 0

