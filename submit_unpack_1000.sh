#!/bin/bash
#SBATCH --job-name=unpack_1000
#SBATCH --output=logs/unpack_1000_%j.out
#SBATCH --error=logs/unpack_1000_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=all

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Load environment
source .venv/bin/activate

# Set Python path
export PYTHONPATH=/mnt/polished-lake/home/cdrobinson/mattergen-interp:$PYTHONPATH

# Run the unpack analysis on 1000 structures from run003
echo "Starting trajectory analysis for 1000 structures..."
python mattergen/scripts/diffusion_step_analysis/unpack_results.py generated_structures/run003 1000

echo "Analysis complete at: $(date)"
echo "Output saved in: generated_structures/run003/plots/"