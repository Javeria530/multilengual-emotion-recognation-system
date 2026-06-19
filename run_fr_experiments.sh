#!/bin/bash -l

#SBATCH -p gpu-v100-16g
#SBATCH --job-name="mtkd-fr-ablations"
#SBATCH --output=%x.out
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --export=HOME,USER,TERM,WRKDIR

module load mamba
module load cuda

source activate /scratch/work/bijoym1/conda_envs/ser_venv

# --- EXPERIMENT 1: Reproduce French Failure Case (Cosine vs Attention) ---
echo "Running FR Split - Cosine Teacher Selection (Baseline)"
python main.py \
    --LEARNING_RATE 3e-5 \
    --BATCH_SIZE 16 \
    --N_EPOCHS 20 \
    --SESSION 1 \
    --TRAINING 1 \
    --PARADIGM "MTKD" \
    --LANGUAGE "FR" \
    --LINGUALITY "Monolingual" \
    --teacher_selection "cosine" \
    --contrastive_weight 0.0

echo "Running FR Split - Attention Teacher Selection"
python main.py \
    --LEARNING_RATE 3e-5 \
    --BATCH_SIZE 16 \
    --N_EPOCHS 20 \
    --SESSION 1 \
    --TRAINING 1 \
    --PARADIGM "MTKD" \
    --LANGUAGE "FR" \
    --LINGUALITY "Monolingual" \
    --teacher_selection "attention" \
    --contrastive_weight 0.0

# --- EXPERIMENT 2: SCL Impact on Minority Class (Anger) Confusion ---
echo "Running FR Split - Attention Teacher Selection WITH Supervised Contrastive Loss (SCL)"
python main.py \
    --LEARNING_RATE 3e-5 \
    --BATCH_SIZE 16 \
    --N_EPOCHS 20 \
    --SESSION 1 \
    --TRAINING 1 \
    --PARADIGM "MTKD" \
    --LANGUAGE "FR" \
    --LINGUALITY "Monolingual" \
    --teacher_selection "attention" \
    --contrastive_weight 0.1 \
    --contrastive_temp 0.07

# Note: The output logs from these runs will now print the full Confusion Matrix
# at the end of every testing epoch. You can directly compare the "Anger" row/column
# between the "Without SCL" (Experiment 1, 2nd run) and "With SCL" (Experiment 2)
# to see if minority-class confusion narrows.
