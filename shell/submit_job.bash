#!/bin/bash
#SBATCH --account=dssc
#SBATCH --job-name=ft_512_cfg_5
#SBATCH --partition=DGX
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40GB
#SBATCH --time=34:00:00

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

# Load CUDA module
# module load cuda/12.1
# The DGX partition already has the correct CUDA version loaded by default
# Load any additional modules if needed

# Activate conda environment
source ~/scratch/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

# Set PyTorch memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the Python job
python "$CODE_DIR/image_generation_emu3.py" \
    --start_idx 2500 \
    --max_samples 2500 \
    --seed 42 \
    --size 512 \
    --batch_size 20 \
    --split validation \
    --guidance_scale 5 \
    --ft_model

# Log end of job
echo "Job completed with exit code $? at $(date)"
