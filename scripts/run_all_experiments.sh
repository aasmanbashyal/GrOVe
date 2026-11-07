#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Experiment pipeline failed at line $LINENO" >&2' ERR

BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

MODE="${1:-full}"
case "$MODE" in
    full|evaluate) ;;
    *) echo "Usage: $0 [full|evaluate]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "Usage: $0 [full|evaluate]" >&2
    exit 2
fi

read -r -a DATASET_NAMES <<< "${DATASETS:-citeseer acm dblp coauthor pubmed amazon}"
read -r -a MODEL_NAMES <<< "${MODELS:-gat gin sage}"
(( ${#DATASET_NAMES[@]} > 0 && ${#MODEL_NAMES[@]} > 0 )) || {
    echo "DATASETS and MODELS must contain at least one name" >&2
    exit 2
}
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-200}"
SPLIT_TYPE="${SPLIT_TYPE:-non-overlapped}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.2}"
VALIDATION_SEED="${VALIDATION_SEED:-42}"

[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be a positive integer" >&2; exit 2; }
[[ "$DEVICE" == cpu || "$DEVICE" == cuda ]] || { echo "DEVICE must be cpu or cuda" >&2; exit 2; }
case "$SPLIT_TYPE" in
    non-overlapped) OVERLAPPED=false ;;
    overlapped) OVERLAPPED=true ;;
    *) echo "SPLIT_TYPE must be non-overlapped or overlapped" >&2; exit 2 ;;
esac
for dataset in "${DATASET_NAMES[@]}"; do
    [[ "$dataset" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid dataset name: $dataset" >&2; exit 2; }
done
for model in "${MODEL_NAMES[@]}"; do
    case "$model" in gat|gin|sage) ;; *) echo "Unknown model: $model" >&2; exit 2 ;; esac
done

run_script() {
    local script="$1"
    shift
    echo "Running $script"
    "$PYTHON_BIN" -m "scripts.$script" "$@"
}

require_file() {
    [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }
}

check_processed_data() {
    local data_dir="$BASE_DIR/data/processed/$SPLIT_TYPE/$1"
    require_file "$data_dir/target_train.pt"
    require_file "$data_dir/test.pt"
    if [[ ! -f "$data_dir/query_train.pt" ]]; then
        require_file "$data_dir/surrogate_train.pt"
    fi
    if [[ ! -f "$data_dir/validation.pt" ]]; then
        require_file "$data_dir/verification.pt"
    fi
}

# Check every requested dataset before starting a potentially long training run.
for dataset in "${DATASET_NAMES[@]}"; do
    if [[ "$MODE" == full ]]; then
        require_file "$BASE_DIR/data/raw/$dataset.npz"
    else
        check_processed_data "$dataset"
        for model in "${MODEL_NAMES[@]}"; do
            require_file "$BASE_DIR/saved_models/$SPLIT_TYPE/${model}_${dataset}/${model}_${dataset}_target_${SPLIT_TYPE}.pt"
        done
    fi
done

mkdir -p "$BASE_DIR"/{saved_models,embeddings,visualizations,results,experiments}

if [[ "$MODE" == full ]]; then
    echo "Phase 1: Preprocessing, target models, independent models, and basic attacks"
    for dataset in "${DATASET_NAMES[@]}"; do
        run_script preprocess_data --dataset "$dataset" --overlapped "$OVERLAPPED" \
            --output-dir "$BASE_DIR/data/processed/$SPLIT_TYPE/$dataset"
        check_processed_data "$dataset"

        for model in "${MODEL_NAMES[@]}"; do
            model_dir="$BASE_DIR/saved_models/$SPLIT_TYPE/${model}_${dataset}"
            embedding_dir="$BASE_DIR/embeddings/$SPLIT_TYPE/${model}_${dataset}"
            mkdir -p "$model_dir" "$embedding_dir"
            training_args=(--model "$model" --dataset "$dataset" --output-dir "$model_dir"
                --embeddings-dir "$embedding_dir" --device "$DEVICE" --epochs "$EPOCHS"
                --split-type "$SPLIT_TYPE" --validation-fraction "$VALIDATION_FRACTION"
                --validation-seed "$VALIDATION_SEED")
            run_script train_model "${training_args[@]}" --model-role target --seed 42
            run_script train_model "${training_args[@]}" --model-role independent \
                --independent-model "$model" --seed 789

            target_model="$model_dir/${model}_${dataset}_target_${SPLIT_TYPE}.pt"
            require_file "$target_model"
            for structure in original idgl; do
                run_script train_stealing_surrogate --target-model-path "$target_model" \
                    --model "$model" --dataset "$dataset" --split-type "$SPLIT_TYPE" \
                    --output-dir "$BASE_DIR/experiments/basic/${model}_${dataset}" \
                    --embeddings-dir "$embedding_dir" --surrogate-architecture "$model" \
                    --recovery-from embedding --structure "$structure" --epochs "$EPOCHS" \
                    --device "$DEVICE" --save-detailed-metrics
                require_file "$embedding_dir/${model}_${dataset}_surrogate_${structure}.pt"
            done
            target_embedding="$embedding_dir/${model}_${dataset}_target.pt"
            independent_embedding="$embedding_dir/${model}_${dataset}_independent_${model}.pt"
            require_file "$target_embedding"
            require_file "$independent_embedding"
            run_script visualize_embeddings --embeddings-path "$target_embedding" \
                "$independent_embedding" "$embedding_dir/${model}_${dataset}_surrogate_original.pt" \
                --output-dir "$BASE_DIR/visualizations/$SPLIT_TYPE/${model}_${dataset}_$(date +%Y%m%d_%H%M%S)" \
                --combined
        done
    done

    echo "Phase 2: Advanced attacks"
    for dataset in "${DATASET_NAMES[@]}"; do
        for model in "${MODEL_NAMES[@]}"; do
            target_model="$BASE_DIR/saved_models/$SPLIT_TYPE/${model}_${dataset}/${model}_${dataset}_target_${SPLIT_TYPE}.pt"
            for attack in fine_tuning double_extraction distribution_shift pruning_0.3 pruning_0.5 pruning_0.7; do
                attack_type="$attack"
                extra_args=()
                if [[ "$attack" == pruning_* ]]; then
                    attack_type=pruning
                    extra_args=(--pruning-ratio "${attack#pruning_}")
                fi
                output_dir="$BASE_DIR/experiments/advanced/${model}_${dataset}/$attack"
                run_script train_stealing_surrogate --target-model-path "$target_model" \
                    --model "$model" --dataset "$dataset" --split-type "$SPLIT_TYPE" \
                    --output-dir "$output_dir" --embeddings-dir "$output_dir/embeddings" \
                    --surrogate-architecture "$model" --recovery-from embedding --structure original \
                    --epochs "$EPOCHS" --device "$DEVICE" --seed 224 \
                    --advanced-attack "$attack_type" "${extra_args[@]}" --save-detailed-metrics
            done
        done
    done

    echo "Phase 3: CSim training and verification"
    csim_dir="$BASE_DIR/experiments/csim/models"
    mkdir -p "$csim_dir"
    for dataset in "${DATASET_NAMES[@]}"; do
        for model in "${MODEL_NAMES[@]}"; do
            embedding_dir="$BASE_DIR/embeddings/$SPLIT_TYPE/${model}_${dataset}"
            run_script train_csim_from_embeddings --model "$model" --dataset "$dataset" \
                --split-type "$SPLIT_TYPE" --embeddings-dir "$BASE_DIR/embeddings" \
                --output-dir "$csim_dir" --device "$DEVICE" --use-grid-search
            for suspect in surrogate_original surrogate_idgl "independent_${model}"; do
                run_script verify_ownership_from_embeddings --target-model-name "${model}_${dataset}_target" \
                    --target-embedding "$embedding_dir/${model}_${dataset}_target.pt" \
                    --suspect-embedding "$embedding_dir/${model}_${dataset}_${suspect}.pt" \
                    --csim-model-dir "$csim_dir" --device "$DEVICE" --threshold 0.5
            done
        done
    done
fi

echo "Phase 4: Comprehensive evaluation"
for dataset in "${DATASET_NAMES[@]}"; do
    run_script all_model_stealing_evaluation --models-dir "$BASE_DIR/saved_models/$SPLIT_TYPE" \
        --dataset "$dataset" --split-type "$SPLIT_TYPE" \
        --output-dir "$BASE_DIR/results/comprehensive_evaluation/$dataset" \
        --device "$DEVICE" --epochs "$EPOCHS" --include-advanced
done

echo "Phase 5: Organizing results"
run_script organize_results --source-dir "$BASE_DIR" --dest-dir "$BASE_DIR/results"
echo "Experiment pipeline completed."
