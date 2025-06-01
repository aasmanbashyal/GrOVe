#!/bin/bash
# This script is used to train the Csim model and verify the ownership of the model
# It is used to verify the ownership of the model by training the Csim model from saved embeddings
# This also prepare separate embeddings for independent and surrogate models
echo "Starting application..."

dataset="citeseer" 
device="cuda"     
model="gat"
split_type="non-overlapped"  
epochs=200
batch_size=100
# Create necessary directories
echo "Creating necessary directories..."
mkdir -p /app/test_csim/embeddings/${split_type}/${model}_${dataset}
mkdir -p /app/test_csim/saved_models/${split_type}/${model}_${dataset}
mkdir -p /app/test_csim/models/csim

echo ""
echo "================================================"
echo "CSIM sample data preparation independent model"
echo "================================================"

INDEPENDENT_EMBEDDING="/app/test_csim/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_independent_${model}.pt"

python3 /app/scripts/train_model.py \
    --model ${model} \
    --dataset ${dataset} \
    --output-dir /app/test_csim/saved_models/${split_type}/${model}_${dataset} \
    --embeddings-dir /app/test_csim/embeddings/${split_type}/${model}_${dataset} \
    --device ${device} \
    --epochs ${epochs} \
    --batch-size ${batch_size} \
    --model-role independent \
    --independent-model ${model} \
    --split-type ${split_type} \
    --seed 789
echo ""
echo "================================================"
echo "CSIM sample data preparation surrogate model"
echo "================================================"

TARGET_MODEL_PATH="/app/saved_models/${split_type}/${model}_${dataset}/${model}_${dataset}_target_${split_type}.pt"

python3 /app/scripts/train_stealing_surrogate.py \
    --target-model-path "$TARGET_MODEL_PATH" \
    --model ${model} \
    --dataset ${dataset} \
    --split-type ${split_type} \
    --output-dir "/app/test_csim/saved_models/${split_type}/${model}_${dataset}" \
    --embeddings-dir "/app/test_csim/embeddings/${split_type}/${model}_${dataset}" \
    --surrogate-architecture ${model} \
    --recovery-from embedding \
    --structure original \
    --epochs 200 \
    --device ${device} \
    --seed 224 \
    --save-detailed-metrics

echo ""
echo "================================================"
echo "CSIM VERIFICATION SYSTEM (EMBEDDING-BASED)"
echo "================================================"

# Check if embeddings exist for training Csim
TARGET_EMBEDDING="/app/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_target.pt"

if [ -f "$TARGET_EMBEDDING" ]; then
    echo "🔍 Training Csim verification system from saved embeddings..."
    
    # Train Csim using saved embeddings
    python3 /app/scripts/train_csim_from_embeddings.py \
        --model ${model} \
        --dataset ${dataset} \
        --split-type ${split_type} \
        --embeddings-dir "/app/embeddings" \
        --output-dir "/app/test_csim/models/csim" \
        --device ${device} \
        --use-grid-search
        
    echo "✅ Csim training completed!"
    
    echo ""
    echo "🔍 Testing ownership verification on different models..."
    
    TARGET_MODEL_NAME="${model}_${dataset}_target"
    
    # Test verification on surrogate models (should detect as surrogate)
    if [ -f "/app/test_csim/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_surrogate_original.pt" ]; then
        echo ""
        echo "--- Verifying Surrogate (Original Structure) ---"
        python3 /app/scripts/verify_ownership_from_embeddings.py \
            --target-model-name "$TARGET_MODEL_NAME" \
            --target-embedding "$TARGET_EMBEDDING" \
            --suspect-embedding "/app/test_csim/embeddings/${split_type}/${model}_${dataset}/${model}_${dataset}_surrogate_original.pt" \
            --csim-model-dir "/app/test_csim/models/csim" \
            --threshold 0.5
    fi
    
    # Test verification on independent model (should detect as independent)
    if [ -f "$INDEPENDENT_EMBEDDING" ]; then
        echo ""
        echo "--- Verifying Independent Model ---"
        python3 /app/scripts/verify_ownership_from_embeddings.py \
            --target-model-name "$TARGET_MODEL_NAME" \
            --target-embedding "$TARGET_EMBEDDING" \
            --suspect-embedding "$INDEPENDENT_EMBEDDING" \
            --csim-model-dir "/app/test_csim/models/csim" \
            --threshold 0.5
    fi
    
    echo ""
    echo "✅ Csim verification testing completed!"
    echo "Results show whether each model is detected as surrogate or independent"
else
    echo "❌ Target embeddings not found at: $TARGET_EMBEDDING"
    echo "Skipping Csim training and verification..."
fi

echo "================================================"