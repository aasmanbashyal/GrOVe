#!/bin/bash

echo "Starting application..."
# echo "Available GPUs (according to nvidia-smi):"
# nvidia-smi || echo "nvidia-smi not found or no GPU detected by it."

dataset="citeseer" 
dataset_list=("citeseer" "acm" "dblp" "coauthor" "pubmed" "amazon")
device="cuda"     
model="gat"
model_list=("gat" "gin" "sage")
independent_model_list=("gat" "gin" "sage")
split_type="non-overlapped"  
epochs=200

for dataset in "${dataset_list[@]}"; do
    for model in "${model_list[@]}"; do
        # Create necessary directories
        echo "Creating directories..."
        mkdir -p /app/data/processed/${split_type}/${dataset}
        mkdir -p /app/saved_models/${split_type}/${model}_${dataset}
        mkdir -p /app/embeddings/${split_type}/${model}_${dataset}
        mkdir -p /app/new_visualizations/${split_type}/${model}_${dataset}
        # mkdir -p /app/models/csim

        echo "Preprocessing data..."
        python3 /app/scripts/preprocess_data.py --dataset ${dataset} --overlapped false --output-dir /app/data/processed/${split_type}/${dataset}

        echo ""
        echo "================================================"
        echo "TRAINING TARGET MODEL (${model} on ${dataset})"
        echo "================================================"
        python3 /app/scripts/train_model.py --model ${model} --dataset ${dataset} --output-dir /app/saved_models/${split_type}/${model}_${dataset} --embeddings-dir /app/embeddings/${split_type}/${model}_${dataset} --device ${device} --epochs ${epochs} --model-role target --split-type ${split_type} --seed 42

        echo ""
        echo "================================================"
        echo "TRAINING INDEPENDENT MODELS (${dataset})"
        echo "================================================"

        for independent_model in ${independent_model_list[@]}; do
            python3 /app/scripts/train_model.py --model ${model} --dataset ${dataset} --output-dir /app/saved_models/${split_type}/${model}_${dataset} --embeddings-dir /app/embeddings/${split_type}/${model}_${dataset} --device ${device} --epochs ${epochs} --model-role independent --independent-model ${independent_model} --split-type ${split_type} --seed 789
        done


        echo ""
        echo "================================================"
        echo "TRAINING MODEL STEALING SURROGATE (${model} on ${dataset})"
        echo "================================================"

        # Path to trained target model
        TARGET_MODEL_PATH="/app/saved_models/${split_type}/${model}_${dataset}/${model}_${dataset}_target_${split_type}.pt"
        echo "Target model path: ${TARGET_MODEL_PATH}"

        # Check if target model exists
        if [ -f "$TARGET_MODEL_PATH" ]; then
            echo "SUCCESS Target model found, proceeding with model stealing attacks..."
            
            echo ""
            echo "================================================"
            echo "TYPE I ATTACK (Original Structure)"
            echo "================================================"
            
            # Type I attack: Use original graph structure
            echo "Running Type I attack with original graph structure..."
            python3 /app/scripts/train_stealing_surrogate.py \
                --target-model-path "$TARGET_MODEL_PATH" \
                --model ${model} \
                --dataset ${dataset} \
                --split-type ${split_type} \
                --output-dir "/app/saved_models/${split_type}/${model}_${dataset}" \
                --embeddings-dir "/app/embeddings/${split_type}/${model}_${dataset}" \
                --surrogate-architecture ${model} \
                --recovery-from embedding \
                --structure original \
                --epochs ${epochs} \
                --device ${device} \
                --save-detailed-metrics
                
            echo "SUCCESS Type I attack completed!"
            echo "Results saved in: /app/saved_models/${split_type}/${model}_${dataset}"
            
            echo ""
            echo "================================================"
            echo "TYPE II ATTACK (IDGL Reconstructed Structure)"
            echo "================================================"
            
            # Type II attack: Use IDGL reconstructed graph structure
            echo "Running Type II attack with IDGL reconstructed structure..."
            python3 /app/scripts/train_stealing_surrogate.py \
                --target-model-path "$TARGET_MODEL_PATH" \
                --model ${model} \
                --dataset ${dataset} \
                --split-type ${split_type} \
                --output-dir "/app/saved_models/${split_type}/${model}_${dataset}" \
                --embeddings-dir "/app/embeddings/${split_type}/${model}_${dataset}" \
                --surrogate-architecture ${model} \
                --recovery-from embedding \
                --structure idgl \
                --epochs ${epochs} \
                --device ${device} \
                --save-detailed-metrics


            echo "SUCCESS Type II (IDGL) attack completed!"
            echo "Results saved in: /app/saved_models/${split_type}/${model}_${dataset}"

        else
            echo " Target model not found at: $TARGET_MODEL_PATH"
            echo "Skipping model stealing attacks..."
        fi


        echo "================================================"

        # Visualize embeddings with multiple perplexity values
        echo "Visualizing embeddings..."
        
        # Comprehensive visualization including stolen surrogate
        python3 /app/scripts/visualize_embeddings.py \
            --embeddings-path \
                "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_target.pt" \
                "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_independent_${model}.pt" \
                "/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_surrogate_original.pt" \
            --output-dir "/app/new_visualizations/${split_type}/${model}_${dataset}" \
            --combined 
    done
done

echo "SUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESS"
echo "🎉 All datasets and models processed successfully!"



