#!/bin/bash
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --exclude=gpu1              # Exclude node gpu1
#SBATCH --partition=all              # Use the GPU partition
#SBATCH --job-name=Image_Processing          # Job name
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --output=job_output_%j.log     # Redirect output to a file with job ID
#SBATCH --error=job_error_%j.log       # Redirect errors to a file with job ID
#SBATCH --no-requeue

# module purge
module load gnu9/9.4.0
module load cuda/10.2

echo "Starting my job..."

# Start timing the process
start_time=$(date +%s)

kwarg_location=$1
path_to_executable="${PWD%/*}/src/pipeline_executable.py"

# ########### PYTHON PROGRAM #############################
# Ensure output directory exists
mkdir -p "${PWD}/output"

# Correct output file name
output_names="${PWD}/output/output_${SLURM_JOB_ID}_$(basename ${kwarg_location})"

# Activate the environment and run the script
source ../.venv/bin/activate
python "$path_to_executable" "$kwarg_location" >> "$output_names" 2>&1 &
wait


# End timing the process
end_time=$(date +%s)
total_time=$(( (end_time - start_time) / 60 ))

# Print the time to complete the process
echo "Total time to complete the job: $total_time minutes"

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: sbatch runner_pipeline.sh /dev/null 2>&1 & disown

exit 0

# ########### TO REMOVE SOME FILES #########################

# To remove files
# ls *.tif
# rm -r temp_*
# rm -r analysis_*
# rm -r slurm* out* temp_* masks_*

# ########### SLURM COMMANDS #########################
# scancel [jobid]
# squeue -u [username]
# squeue
