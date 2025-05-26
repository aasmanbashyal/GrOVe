#!/bin/bash

echo "Starting application..."
# echo "Available GPUs (according to nvidia-smi):"
# nvidia-smi || echo "nvidia-smi not found or no GPU detected by it."

dataset="citeseer"
device="cuda"     
model="gat"
split_type="non-overlapped"

# Create necessary directories
echo "Creating directories..."
mkdir -p /app/data/processed/${split_type}/${dataset}
mkdir -p /app/models/${split_type}/${model}_${dataset}
mkdir -p /app/embeddings/${split_type}/${model}_${dataset}
mkdir -p /app/visualizations/${split_type}/${model}_${dataset}
# Preprocess data
echo "Preprocessing data..."
python3 /app/scripts/preprocess_data.py --dataset ${dataset} --overlapped false --output-dir /app/data/processed/${split_type}/${dataset}

echo "Training models for ${split_type} splits..."

# Target model with seed 42 
python3 /app/scripts/train_model.py --model ${model} --dataset ${dataset} --output-dir /app/models/${split_type}/${model}_${dataset} --embeddings-dir /app/embeddings/${split_type}/${model}_${dataset} --device ${device} --epochs 200 --model-role target --split-type ${split_type} --seed 42

# Independent model with different seed (789) but same data
python3 /app/scripts/train_model.py --model ${model} --dataset ${dataset} --output-dir /app/models/${split_type}/${model}_${dataset} --embeddings-dir /app/embeddings/${split_type}/${model}_${dataset} --device ${device} --epochs 200 --model-role independent --split-type ${split_type} --seed 789

# Surrogate model
python3 /app/scripts/train_model.py --model ${model} --dataset ${dataset} --output-dir /app/models/${split_type}/${model}_${dataset} --embeddings-dir /app/embeddings/${split_type}/${model}_${dataset} --device ${device} --epochs 200 --model-role surrogate --split-type ${split_type} --seed 42

# Visualize embeddings with multiple perplexity values
echo "Visualizing embeddings..."

python3 /app/scripts/visualize_embeddings.py \
    --embeddings-path "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_target.pt" \
    "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_independent.pt" \
    "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_surrogate.pt" \
    --output-dir "/app/visualizations/${split_type}/${model}_${dataset}" \
    --combined \
    --try-multiple-perplexity \
    --perplexity-values 5 30 50 

echo "Done"
echo "================================================"
