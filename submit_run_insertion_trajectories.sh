#!/bin/bash
#SBATCH --job-name=run_insertion_trajectories
#SBATCH --output=jobs/run_insertion_trajectories/job_%j.out
#SBATCH --error=jobs/run_insertion_trajectories/job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=88G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90    # When to send email notifications
#SBATCH --mail-user=connietherobin@gmail.com          # Your email address
# Activate your Python environment
source .venv/bin/activate
# Set environment variables for your authentication tokens
export WANDB_API_KEY="[REMOVED]"
export HF_TOKEN="[REMOVED]"
# Run your script
python mattergen/scripts/run_insertion_trajectories_2.py