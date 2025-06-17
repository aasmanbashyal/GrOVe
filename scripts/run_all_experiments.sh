#!/bin/bash

echo "======================================================================"


DATASETS=("citeseer" "acm" "dblp" "coauthor" "pubmed" "amazon")

MODELS=("gat" "gin" "sage")

DEVICE="cuda"
EPOCHS=200
SPLIT_TYPE="non-overlapped"

if [ -d "/app" ]; then
    BASE_DIR="/app"
    echo "Running in Docker environment"
else
    BASE_DIR="$(pwd)"
    echo "Running in native environment: $BASE_DIR"
fi

echo "Creating directory structure..."
mkdir -p ${BASE_DIR}/{data,saved_models,embeddings,visualizations,results}
mkdir -p ${BASE_DIR}/results/{basic_attacks,advanced_attacks,csim_verification,comprehensive_evaluation}
mkdir -p ${BASE_DIR}/experiments/{basic,advanced,csim}

echo ""
echo "Fixed directory structure created!"
echo "======================================================================"

# Phase 1: Basic Experimental Pipeline
echo ""
echo "PROCESSING Phase 1: Running Basic Experimental Pipeline"
echo "======================================================================"

# for dataset in "${DATASETS[@]}"; do
#     for model in "${MODELS[@]}"; do
#         echo ""
#         echo "Processing: $model on $dataset"
#         echo "----------------------------------------"
        
#         mkdir -p ${BASE_DIR}/data/processed/${SPLIT_TYPE}/${dataset}
#         mkdir -p ${BASE_DIR}/saved_models/${SPLIT_TYPE}/${model}_${dataset}
#         mkdir -p ${BASE_DIR}/embeddings/${SPLIT_TYPE}/${model}_${dataset}
#         mkdir -p ${BASE_DIR}/visualizations/${SPLIT_TYPE}/${model}_${dataset}

#         # 1. Data preprocessing
#         echo "Preprocessing data for $dataset..."
#         python3 ${BASE_DIR}/scripts/preprocess_data.py \
#             --dataset ${dataset} \
#             --overlapped false \
#             --output-dir ${BASE_DIR}/data/processed/${SPLIT_TYPE}/${dataset}

#         # 2. Train target model
#         echo "Training target model ($model on $dataset)..."
#         python3 ${BASE_DIR}/scripts/train_model.py \
#             --model ${model} \
#             --dataset ${dataset} \
#             --output-dir ${BASE_DIR}/saved_models/${SPLIT_TYPE}/${model}_${dataset} \
#             --embeddings-dir ${BASE_DIR}/embeddings/${SPLIT_TYPE}/${model}_${dataset} \
#             --device ${DEVICE} \
#             --epochs ${EPOCHS} \
#             --model-role target \
#             --split-type ${SPLIT_TYPE} \
#             --seed 42

#         # 3. Train independent models
#         echo "PROCESSING Training independent models for $dataset..."
#         for independent_model in "${MODELS[@]}"; do
#             python3 ${BASE_DIR}/scripts/train_model.py \
#                 --model ${model} \
#                 --dataset ${dataset} \
#                 --output-dir ${BASE_DIR}/saved_models/${SPLIT_TYPE}/${model}_${dataset} \
#                 --embeddings-dir ${BASE_DIR}/embeddings/${SPLIT_TYPE}/${model}_${dataset} \
#                 --device ${DEVICE} \
#                 --epochs ${EPOCHS} \
#                 --model-role independent \
#                 --independent-model ${independent_model} \
#                 --split-type ${SPLIT_TYPE} \
#                 --seed 789
#         done

#         # 4. Basic model stealing attacks (consistent paths)
#         TARGET_MODEL_PATH="${BASE_DIR}/saved_models/${SPLIT_TYPE}/${model}_${dataset}/${model}_${dataset}_target_${SPLIT_TYPE}.pt"
        
#         if [ -f "$TARGET_MODEL_PATH" ]; then
#             echo "Running basic model stealing attacks..."
            
#             # Type I attack (Original structure)
#             python3 ${BASE_DIR}/scripts/train_stealing_surrogate.py \
#                 --target-model-path "$TARGET_MODEL_PATH" \
#                 --model ${model} \
#                 --dataset ${dataset} \
#                 --split-type ${SPLIT_TYPE} \
#                 --output-dir "${BASE_DIR}/experiments/basic/${model}_${dataset}" \
#                 --embeddings-dir "${BASE_DIR}/embeddings/${SPLIT_TYPE}/${model}_${dataset}" \
#                 --surrogate-architecture ${model} \
#                 --recovery-from embedding \
#                 --structure original \
#                 --epochs ${EPOCHS} \
#                 --device ${DEVICE} \
#                 --save-detailed-metrics
                
#             # Type II attack (IDGL structure)
#             python3 ${BASE_DIR}/scripts/train_stealing_surrogate.py \
#                 --target-model-path "$TARGET_MODEL_PATH" \
#                 --model ${model} \
#                 --dataset ${dataset} \
#                 --split-type ${SPLIT_TYPE} \
#                 --output-dir "${BASE_DIR}/experiments/basic/${model}_${dataset}" \
#                 --embeddings-dir "${BASE_DIR}/embeddings/${SPLIT_TYPE}/${model}_${dataset}" \
#                 --surrogate-architecture ${model} \
#                 --recovery-from embedding \
#                 --structure idgl \
#                 --epochs ${EPOCHS} \
#                 --device ${DEVICE} \
#                 --save-detailed-metrics

#             # Visualize embeddings with correct paths and checks
#             TARGET_EMB="${BASE_DIR}/embeddings/${SPLIT_TYPE}/${model}_${dataset}/${model}_${dataset}_target.pt"
#             INDEPENDENT_EMB="${BASE_DIR}/embeddings/${SPLIT_TYPE}/${model}_${dataset}/${model}_${dataset}_independent_${model}.pt"
#             SURROGATE_EMB="${BASE_DIR}/embeddings/${SPLIT_TYPE}/${model}_${dataset}/${model}_${dataset}_surrogate_original.pt"
            
#             echo "Generating visualizations..."
#             if [ -f "$TARGET_EMB" ] && [ -f "$INDEPENDENT_EMB" ] && [ -f "$SURROGATE_EMB" ]; then
#                 # Create unique visualization directory to avoid overrides
#                 VIZ_DIR="${BASE_DIR}/visualizations/${SPLIT_TYPE}/${model}_${dataset}_$(date +%Y%m%d_%H%M%S)"
#                 mkdir -p "$VIZ_DIR"
                
#                 python3 ${BASE_DIR}/scripts/visualize_embeddings.py \
#                     --embeddings-path "$TARGET_EMB" "$INDEPENDENT_EMB" "$SURROGATE_EMB" \
#                     --output-dir "$VIZ_DIR" \
#                     --combined
                
#                 echo "Visualizations saved to: $VIZ_DIR"
#             else
#                 echo "Some embeddings not found, skipping visualization:"
#                 echo "    Target: $([ -f "$TARGET_EMB" ] && echo "" || echo "❌") $TARGET_EMB"
#                 echo "    Independent: $([ -f "$INDEPENDENT_EMB" ] && echo "" || echo "❌") $INDEPENDENT_EMB"
#                 echo "    Surrogate: $([ -f "$SURROGATE_EMB" ] && echo "" || echo "❌") $SURROGATE_EMB"
#             fi
#         else
#             echo "Target model not found: $TARGET_MODEL_PATH"
#         fi
#     done
# done

# # Phase 2: Advanced Attacks
# echo ""
# echo "PROCESSING Phase 2: Running Advanced Attacks"
# echo "======================================================================"

# for dataset in "${DATASETS[@]}"; do
#     for model in "${MODELS[@]}"; do
#         # Use consistent path resolution
#         TARGET_MODEL_PATH="${BASE_DIR}/saved_models/${SPLIT_TYPE}/${model}_${dataset}/${model}_${dataset}_target_${SPLIT_TYPE}.pt"
        
#         if [ -f "$TARGET_MODEL_PATH" ]; then
#             echo ""
#             echo "Advanced attacks: $model on $dataset"
#             echo "----------------------------------------"
            
#             # Create advanced experiment directories
#             mkdir -p ${BASE_DIR}/experiments/advanced/${model}_${dataset}
            
#             # Fine-tuning attack
#             echo " Fine-tuning attack..."
#             mkdir -p "${BASE_DIR}/experiments/advanced/${model}_${dataset}/fine_tuning/embeddings"
#             if python3 ${BASE_DIR}/scripts/train_stealing_surrogate.py \
#                 --target-model-path "$TARGET_MODEL_PATH" \
#                 --model ${model} \
#                 --dataset ${dataset} \
#                 --split-type ${SPLIT_TYPE} \
#                 --output-dir "${BASE_DIR}/experiments/advanced/${model}_${dataset}/fine_tuning" \
#                 --embeddings-dir "${BASE_DIR}/experiments/advanced/${model}_${dataset}/fine_tuning/embeddings" \
#                 --surrogate-architecture ${model} \
#                 --recovery-from embedding \
#                 --structure original \
#                 --epochs ${EPOCHS} \
#                 --device ${DEVICE} \
#                 --seed 224 \
#                 --advanced-attack fine_tuning \
#                 --save-detailed-metrics; then
#                 echo " Fine-tuning attack completed"
#             else
#                 echo " Fine-tuning attack failed"
#             fi

#             # Double extraction attack
#             echo "Double extraction attack..."
#             mkdir -p "${BASE_DIR}/experiments/advanced/${model}_${dataset}/double_extraction/embeddings"
#             if python3 ${BASE_DIR}/scripts/train_stealing_surrogate.py \
#                 --target-model-path "$TARGET_MODEL_PATH" \
#                 --model ${model} \
#                 --dataset ${dataset} \
#                 --split-type ${SPLIT_TYPE} \
#                 --output-dir "${BASE_DIR}/experiments/advanced/${model}_${dataset}/double_extraction" \
#                 --embeddings-dir "${BASE_DIR}/experiments/advanced/${model}_${dataset}/double_extraction/embeddings" \
#                 --surrogate-architecture ${model} \
#                 --recovery-from embedding \
#                 --structure original \
#                 --epochs ${EPOCHS} \
#                 --device ${DEVICE} \
#                 --seed 224 \
#                 --advanced-attack double_extraction \
#                 --save-detailed-metrics; then
#                 echo " Double extraction attack completed"
#             else
#                 echo " Double extraction attack failed"
#             fi

#             # Distribution shift attack
#             echo "Distribution shift attack..."
#             mkdir -p "${BASE_DIR}/experiments/advanced/${model}_${dataset}/distribution_shift/embeddings"
#             if python3 ${BASE_DIR}/scripts/train_stealing_surrogate.py \
#                 --target-model-path "$TARGET_MODEL_PATH" \
#                 --model ${model} \
#                 --dataset ${dataset} \
#                 --split-type ${SPLIT_TYPE} \
#                 --output-dir "${BASE_DIR}/experiments/advanced/${model}_${dataset}/distribution_shift" \
#                 --embeddings-dir "${BASE_DIR}/experiments/advanced/${model}_${dataset}/distribution_shift/embeddings" \
#                 --surrogate-architecture ${model} \
#                 --recovery-from embedding \
#                 --structure original \
#                 --epochs ${EPOCHS} \
#                 --device ${DEVICE} \
#                 --seed 224 \
#                 --advanced-attack distribution_shift \
#                 --save-detailed-metrics; then
#                 echo "Distribution shift attack completed"
#             else
#                 echo "Distribution shift attack failed"
#             fi

#             # Pruning attacks with different ratios
#             for pruning_ratio in 0.3 0.5 0.7; do
#                 echo "Pruning attack (ratio: $pruning_ratio)..."
#                 mkdir -p "${BASE_DIR}/experiments/advanced/${model}_${dataset}/pruning_${pruning_ratio}/embeddings"
#                 if python3 ${BASE_DIR}/scripts/train_stealing_surrogate.py \
#                     --target-model-path "$TARGET_MODEL_PATH" \
#                     --model ${model} \
#                     --dataset ${dataset} \
#                     --split-type ${SPLIT_TYPE} \
#                     --output-dir "${BASE_DIR}/experiments/advanced/${model}_${dataset}/pruning_${pruning_ratio}" \
#                     --embeddings-dir "${BASE_DIR}/experiments/advanced/${model}_${dataset}/pruning_${pruning_ratio}/embeddings" \
#                     --surrogate-architecture ${model} \
#                     --recovery-from embedding \
#                     --structure original \
#                     --epochs ${EPOCHS} \
#                     --device ${DEVICE} \
#                     --seed 224 \
#                     --advanced-attack pruning \
#                     --pruning-ratio ${pruning_ratio} \
#                     --save-detailed-metrics; then
#                     echo "Pruning attack (${pruning_ratio}) completed"
#                 else
#                     echo "Pruning attack (${pruning_ratio}) failed"
#                 fi
#             done
#         fi
#     done
# done

# # Phase 3: CSim Training and Verification
# echo ""
# echo "PROCESSING Phase 3: CSim Training and Verification"
# echo "======================================================================"

# # Create CSim directories
# mkdir -p ${BASE_DIR}/experiments/csim/models

# for dataset in "${DATASETS[@]}"; do
#     for model in "${MODELS[@]}"; do
#         TARGET_EMBEDDING="${BASE_DIR}/embeddings/${SPLIT_TYPE}/${model}_${dataset}/${model}_${dataset}_target.pt"
        
#         if [ -f "$TARGET_EMBEDDING" ]; then
#             echo ""
#             echo "DEBUG CSim training: $model on $dataset"
#             echo "----------------------------------------"
            
#             # Train CSim
#             python3 ${BASE_DIR}/scripts/train_csim_from_embeddings.py \
#                 --model ${model} \
#                 --dataset ${dataset} \
#                 --split-type ${SPLIT_TYPE} \
#                 --embeddings-dir "${BASE_DIR}/embeddings" \
#                 --output-dir "${BASE_DIR}/experiments/csim/models" \
#                 --device ${DEVICE} \
#                 --use-grid-search
                
#             # Test verification
#             TARGET_MODEL_NAME="${model}_${dataset}_target"
            
#             # Test on surrogate models
#             for surrogate_type in "original" "idgl"; do
#                 SURROGATE_EMBEDDING="${BASE_DIR}/embeddings/${SPLIT_TYPE}/${model}_${dataset}/${model}_${dataset}_surrogate_${surrogate_type}.pt"
#                 if [ -f "$SURROGATE_EMBEDDING" ]; then
#                     echo "DEBUG Testing verification on surrogate ($surrogate_type)..."
#                     python3 ${BASE_DIR}/scripts/verify_ownership_from_embeddings.py \
#                         --target-model-name "$TARGET_MODEL_NAME" \
#                         --target-embedding "$TARGET_EMBEDDING" \
#                         --suspect-embedding "$SURROGATE_EMBEDDING" \
#                         --csim-model-dir "${BASE_DIR}/experiments/csim/models" \
#                         --threshold 0.5
#                 fi
#             done
#         fi
#     done
# done

# Phase 4: Comprehensive Evaluation
echo ""
echo "PROCESSING Phase 4: Comprehensive Evaluation and Results Generation"
echo "======================================================================"

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Running comprehensive evaluation for $dataset..."
    echo "----------------------------------------"
    
    # Use consistent paths
    python3 ${BASE_DIR}/scripts/all_model_stealing_evaluation.py \
        --models-dir "${BASE_DIR}/saved_models/${SPLIT_TYPE}" \
        --dataset ${dataset} \
        --split-type ${SPLIT_TYPE} \
        --output-dir "${BASE_DIR}/results/comprehensive_evaluation/${dataset}" \
        --device ${DEVICE} \
        --epochs ${EPOCHS} \
        --include-advanced
done

# Phase 5: Results Organization and Summary
echo ""
echo "PROCESSING Phase 5: Organizing and Summarizing Results"
echo "======================================================================"

# Step 5: Organize all experimental results
echo " Organizing experimental results..."
if python3 ${BASE_DIR}/scripts/organize_results.py --source-dir ${BASE_DIR} --dest-dir ${BASE_DIR}/results; then
    echo "Results organization completed"
else
    echo "Results organization failed"
fi


echo "======================================================================" 