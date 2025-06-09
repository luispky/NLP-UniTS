#!/bin/bash
#SBATCH --account=dssc
#SBATCH --job-name=coco-30k
#SBATCH --partition=DGX
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=04:00:00

# Conda environment name
CONDA_ENV_NAME=emu3

# Project directories
PROJECT_ROOT="/u/dssc/lpalaciosflores/NLP-UniTS"
CODE_DIR="$PROJECT_ROOT/code"

# Job metadata logging
echo '------------------------------------------------------'
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_JOB_CPUS_PER_NODE"
echo "Running on node(s): $SLURM_JOB_NODELIST"
echo "Started at: $(date)"
echo '------------------------------------------------------'

# Activate conda environment
source ~/scratch/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

# Run the Python job to download the restval dataset
python "$CODE_DIR/download_COCOKarpathyDataset.py" --split restval

# Log end of job
echo "Job completed with exit code $? at $(date)"
