#!/bin/bash

echo "Starting application..."

dataset="citeseer" 
device="cuda"     
model="gat"
split_type="non-overlapped"  
epochs=200
batch_size=1024


TARGET_MODEL_PATH="/app/saved_models/${split_type}/${model}_${dataset}/${model}_${dataset}_target_${split_type}.pt"

# Check if target model exists
if [ ! -f "$TARGET_MODEL_PATH" ]; then
    echo " Error: Target model not found at: $TARGET_MODEL_PATH"
    echo "Please ensure the target model is trained first."
    exit 1
fi

echo ""
echo "================================================"
echo "ADVANCED ATTACK: Fine-tuning"
echo "================================================"

python3 /app/scripts/train_stealing_surrogate.py \
    --target-model-path "$TARGET_MODEL_PATH" \
    --model ${model} \
    --dataset ${dataset} \
    --split-type ${split_type} \
    --output-dir "/app/test_advance/saved_models/${split_type}/${model}_${dataset}_fine_tuning" \
    --embeddings-dir "/app/test_advance/embeddings/${split_type}/${model}_${dataset}_fine_tuning" \
    --surrogate-architecture ${model} \
    --recovery-from embedding \
    --structure original \
    --epochs 200 \
    --device ${device} \
    --seed 224 \
    --advanced-attack fine_tuning \
    --save-detailed-metrics

check_command "Fine-tuning attack training"

python3 /app/scripts/visualize_embeddings.py \
    --embeddings-path \
       "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_target.pt" \
       "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_independent_${model}.pt" \
       "/app/test_advance/embeddings/${split_type}/${model}_${dataset}_fine_tuning/${model}_${dataset}_surrogate_original_fine_tuning.pt" \
    --output-dir "/app/test_advance/new_visualizations/${split_type}/${model}_${dataset}_fine_tuning" \
    --combined

check_command "Fine-tuning visualization"

echo ""
echo "================================================"
echo "ADVANCED ATTACK: Double extraction"
echo "================================================"

python3 /app/scripts/train_stealing_surrogate.py \
    --target-model-path "$TARGET_MODEL_PATH" \
    --model ${model} \
    --dataset ${dataset} \
    --split-type ${split_type} \
    --output-dir "/app/test_advance/saved_models/${split_type}/${model}_${dataset}_double_extraction_type2" \
    --embeddings-dir "/app/test_advance/embeddings/${split_type}/${model}_${dataset}_double_extraction_type2" \
    --surrogate-architecture ${model} \
    --recovery-from embedding \
    --structure idgl \
    --epochs 200 \
    --device ${device} \
    --seed 224 \
    --advanced-attack double_extraction \
    --save-detailed-metrics

check_command "Double extraction attack training"

python3 /app/scripts/visualize_embeddings.py \
    --embeddings-path \
       "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_target.pt" \
       "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_independent_${model}.pt" \
       "/app/test_advance/embeddings/${split_type}/${model}_${dataset}_double_extraction_type2/${model}_${dataset}_surrogate_idgl_double_extraction.pt" \
    --output-dir "/app/test_advance/new_visualizations/${split_type}/${model}_${dataset}_double_extraction_type2" \
    --combined

check_command "Double extraction visualization"

echo ""
echo "================================================"
echo "ADVANCED ATTACK: Distribution shift"
echo "================================================"

python3 /app/scripts/train_stealing_surrogate.py \
    --target-model-path "$TARGET_MODEL_PATH" \
    --model ${model} \
    --dataset ${dataset} \
    --split-type ${split_type} \
    --output-dir "/app/test_advance/saved_models/${split_type}/${model}_${dataset}_distribution_shift" \
    --embeddings-dir "/app/test_advance/embeddings/${split_type}/${model}_${dataset}_distribution_shift" \
    --surrogate-architecture ${model} \
    --recovery-from embedding \
    --structure original \
    --epochs 200 \
    --device ${device} \
    --seed 224 \
    --advanced-attack distribution_shift \
    --save-detailed-metrics

check_command "Distribution shift attack training"

python3 /app/scripts/visualize_embeddings.py \
    --embeddings-path \
       "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_target.pt" \
       "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_independent_${model}.pt" \
       "/app/test_advance/embeddings/${split_type}/${model}_${dataset}_distribution_shift/${model}_${dataset}_surrogate_original_distribution_shift.pt" \
    --output-dir "/app/test_advance/new_visualizations/${split_type}/${model}_${dataset}_distribution_shift" \
    --combined

echo ""
echo "================================================"
echo "ADVANCED ATTACK: Double extraction"
echo "================================================"

python3 /app/scripts/train_stealing_surrogate.py \
    --target-model-path "$TARGET_MODEL_PATH" \
    --model ${model} \
    --dataset ${dataset} \
    --split-type ${split_type} \
    --output-dir "/app/test_advance/saved_models/${split_type}/${model}_${dataset}_double_extraction_type1" \
    --embeddings-dir "/app/test_advance/embeddings/${split_type}/${model}_${dataset}_double_extraction_type1" \
    --surrogate-architecture ${model} \
    --recovery-from embedding \
    --structure original \
    --epochs 200 \
    --device ${device} \
    --seed 224 \
    --advanced-attack double_extraction \
    --save-detailed-metrics

check_command "Double extraction attack training"

python3 /app/scripts/visualize_embeddings.py \
    --embeddings-path \
       "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_target.pt" \
       "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_independent_${model}.pt" \
       "/app/test_advance/embeddings/${split_type}/${model}_${dataset}_double_extraction_type1/${model}_${dataset}_surrogate_original_double_extraction.pt" \
    --output-dir "/app/test_advance/new_visualizations/${split_type}/${model}_${dataset}_double_extraction_type1" \
    --combined

check_command "Double extraction visualization"


# Pruning ratio 0.3
echo "Training with pruning ratio 0.3..."
python3 /app/scripts/train_stealing_surrogate.py \
    --target-model-path "$TARGET_MODEL_PATH" \
    --model ${model} \
    --dataset ${dataset} \
    --split-type ${split_type} \
    --output-dir "/app/test_advance/saved_models/${split_type}/${model}_${dataset}_pruning_03" \
    --embeddings-dir "/app/test_advance/embeddings/${split_type}/${model}_${dataset}_pruning_03" \
    --surrogate-architecture ${model} \
    --recovery-from embedding \
    --structure original \
    --epochs 200 \
    --device ${device} \
    --seed 224 \
    --advanced-attack pruning \
    --pruning-ratio 0.3 \
    --save-detailed-metrics

check_command "Pruning 0.3 training"

python3 /app/scripts/visualize_embeddings.py \
    --embeddings-path \
       "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_target.pt" \
       "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_independent_${model}.pt" \
       "/app/test_advance/embeddings/${split_type}/${model}_${dataset}_pruning_03/${model}_${dataset}_surrogate_original_pruning.pt" \
    --output-dir "/app/test_advance/new_visualizations/${split_type}/${model}_${dataset}_pruning_03" \
    --combined

check_command "Pruning 0.3 visualization"



echo "================================================"