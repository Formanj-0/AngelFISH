#!/bin/bash
# runner_pipeline.sh

#SBATCH --gres=gpu:1                 # Request 1 GPU if available
#SBATCH --partition=all              # Use all partitions
#SBATCH --job-name=ERon              # Job name
#SBATCH --ntasks=64                  # Number of tasks
#SBATCH --output=job_output_%j.log   # STDOUT → this file
#SBATCH --error=job_error_%j.log     # STDERR → this file
#SBATCH --requeue

module reset
module load gnu13/13.2.0

# Set CUDA paths (harmless if CUDA not installed)
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

echo "Checking for GPU..."
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    echo "-> GPU detected!"
    export DEVICE="cuda"
else
    echo "-> No GPU available; falling back to CPU."
    export DEVICE="cpu"
fi
echo "Using DEVICE=$DEVICE"
echo "Starting job..."

start_time=$(date +%s)

kwarg_location=$1
path_to_executable="${PWD%/*}/src/pipeline_executable2.py"

# Ensure output dir exists
mkdir -p "${PWD}/output"
output_file="${PWD}/output/output__$(basename "${kwarg_location}")"

# Activate your virtualenv
source ../.venv/bin/activate

# Run in foreground, passing through our device choice
python "$path_to_executable" \
    "$kwarg_location" \
    --device "$DEVICE" \
    >> "$output_file" 2>&1

end_time=$(date +%s)
total_time=$(( (end_time - start_time) / 60 ))
echo "Total time: $total_time minutes"

exit 0